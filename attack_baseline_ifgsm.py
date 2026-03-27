#!/usr/bin/env python3
"""
Standard masked I-FGSM adversarial attack for YOLOv8 license-plate detection
(using ground-truth XML bounding boxes) — no EOT.

Pipeline
--------
- Reads GT bbox from Pascal VOC-style XML annotations
- Letterboxes input images to `imgsz` while preserving resize/padding metadata
- Perturbs only inside the GT bbox mask (in letterbox space), with optional margin
- Uses a differentiable proxy loss on YOLO raw head outputs (top-k mean score)
- Performs early stopping when detection fails on the reconstructed original-resolution image
- Saves:
    * clean detection visualization
    * clean/adversarial letterboxed and original-resolution images
    * GT mask, masked overlays, and bbox crops
    * plate patch crops (+ optional resized printable patch)
    * per-image summary.txt
- Writes CSV: psnr_masked.csv

Important note
--------------
This attack suppresses detector confidence by minimizing a differentiable confidence
proxy derived from YOLO raw outputs. The update therefore uses gradient descent on
the proxy within an L_inf perturbation budget.
"""

import os
import argparse
import csv
from pathlib import Path
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import random

from ultralytics import YOLO
import torch


# ========================= Reproducibility ========================= #
def seed_all(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Best-effort reproducibility
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ========================= I/O helpers ========================= #
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_rgb(path: str, rgb: np.ndarray):
    ensure_dir(os.path.dirname(path) or ".")
    cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def save_gray(path: str, gray: np.ndarray):
    ensure_dir(os.path.dirname(path) or ".")
    cv2.imwrite(path, gray)


def validate_inputs(weights: str, source_dir: str, xml_dir: str):
    if not Path(weights).exists():
        raise FileNotFoundError(f"Weights file not found: {weights}")
    if not Path(source_dir).exists():
        raise FileNotFoundError(f"Source image directory not found: {source_dir}")
    if not Path(xml_dir).exists():
        raise FileNotFoundError(f"XML annotation directory not found: {xml_dir}")


# =================== Letterbox helpers =================== #
def letterbox_meta(im_rgb: np.ndarray, imgsz: int):
    h, w = im_rgb.shape[:2]
    r = min(imgsz / h, imgsz / w)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(im_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w, pad_h = imgsz - new_w, imgsz - new_h
    left, right = pad_w // 2, pad_w - pad_w // 2
    top, bottom = pad_h // 2, pad_h - pad_h // 2

    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        borderType=cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )

    return padded, {
        "orig_h": h,
        "orig_w": w,
        "r": r,
        "new_h": new_h,
        "new_w": new_w,
        "top": top,
        "left": left,
    }


def letterbox_to_tensor(path: str, imgsz: int, device: str):
    im_bgr = cv2.imread(path)
    if im_bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
    padded_rgb, meta = letterbox_meta(im_rgb, imgsz)
    x = torch.from_numpy(padded_rgb).permute(2, 0, 1).float() / 255.0
    return x.unsqueeze(0).to(device), meta, im_rgb, padded_rgb


def tensor_to_rgb_uint8(x01: torch.Tensor) -> np.ndarray:
    x = x01.detach().clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
    return (x * 255).round().astype(np.uint8)


def reconstruct_orig_from_letterboxed_rgb(letter_rgb: np.ndarray, meta: dict) -> np.ndarray:
    top, left = int(meta["top"]), int(meta["left"])
    new_h, new_w = int(meta["new_h"]), int(meta["new_w"])
    orig_h, orig_w = int(meta["orig_h"]), int(meta["orig_w"])

    cropped = letter_rgb[top:top + new_h, left:left + new_w]
    if cropped.size == 0:
        return cv2.resize(letter_rgb, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    return cv2.resize(cropped, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)


# =================== Mask + crop helpers =================== #
def crop_with_margin(rgb: np.ndarray, xyxy, margin: int):
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = map(int, xyxy)

    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(w, x2 + margin)
    y2 = min(h, y2 + margin)

    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)

    return rgb[y1:y2, x1:x2].copy(), [x1, y1, x2, y2]


def make_bbox_mask(imgsz: int, xyxy, margin: int, device: str):
    x1, y1, x2, y2 = map(int, xyxy)

    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(imgsz, x2 + margin)
    y2 = min(imgsz, y2 + margin)

    if x2 <= x1:
        x2 = min(imgsz, x1 + 1)
    if y2 <= y1:
        y2 = min(imgsz, y1 + 1)

    m = torch.zeros((1, 1, imgsz, imgsz), device=device)
    m[:, :, y1:y2, x1:x2] = 1.0
    return m


def save_gt_masked_area_outputs(per_dir: str, im_rgb: np.ndarray, adv_rgb: np.ndarray, bbox_xyxy, margin: int):
    """
    Saves:
      - gt_mask_orig.png          : binary mask at original resolution
      - clean_masked_overlay.png  : clean image with bbox region kept, rest black
      - adv_masked_overlay.png    : adversarial image with bbox region kept, rest black
      - clean_bbox_crop.png       : cropped bbox+margin from clean image
      - adv_bbox_crop.png         : cropped bbox+margin from adversarial image
    """
    h, w = im_rgb.shape[:2]
    x1, y1, x2, y2 = map(int, bbox_xyxy)

    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(w, x2 + margin)
    y2 = min(h, y2 + margin)

    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    save_gray(os.path.join(per_dir, "gt_mask_orig.png"), mask)

    clean_overlay = np.zeros_like(im_rgb)
    adv_overlay = np.zeros_like(adv_rgb)
    clean_overlay[y1:y2, x1:x2] = im_rgb[y1:y2, x1:x2]
    adv_overlay[y1:y2, x1:x2] = adv_rgb[y1:y2, x1:x2]

    save_rgb(os.path.join(per_dir, "clean_masked_overlay.png"), clean_overlay)
    save_rgb(os.path.join(per_dir, "adv_masked_overlay.png"), adv_overlay)

    clean_crop = im_rgb[y1:y2, x1:x2].copy()
    adv_crop = adv_rgb[y1:y2, x1:x2].copy()
    save_rgb(os.path.join(per_dir, "clean_bbox_crop.png"), clean_crop)
    save_rgb(os.path.join(per_dir, "adv_bbox_crop.png"), adv_crop)

    with open(os.path.join(per_dir, "gt_mask_info.txt"), "w", encoding="utf-8") as f:
        f.write(f"bbox_xyxy_xml: {bbox_xyxy}\n")
        f.write(f"margin_px: {margin}\n")
        f.write(f"mask_xyxy_used: {[x1, y1, x2, y2]}\n")
        f.write(f"image_shape: {(h, w)}\n")


# =================== Metrics =================== #
def psnr_masked(clean: torch.Tensor, adv: torch.Tensor, mask_1ch: torch.Tensor, eps: float = 1e-12):
    mask3 = mask_1ch.repeat(1, 3, 1, 1)
    diff2 = (clean - adv) ** 2
    mse = (diff2 * mask3).sum() / mask3.sum().clamp_min(1.0)
    return 10.0 * torch.log10(1.0 / mse.clamp_min(eps))


# =================== Loss proxy =================== #
def confidence_proxy(raw, topk: int = 200):
    """
    Differentiable proxy that increases when YOLO raw head scores increase.
    This is not the same as post-NMS confidence from model.predict().
    """
    pred = raw[0] if isinstance(raw, (list, tuple)) else raw

    if torch.is_tensor(pred) and pred.ndim >= 2 and pred.shape[-1] >= 6:
        obj = pred[..., 4].sigmoid()
        cls = pred[..., 5:].sigmoid().max(dim=-1).values
        score = (obj * cls).reshape(-1)
    else:
        if not torch.is_tensor(pred):
            pred = torch.as_tensor(pred)
        score = pred.reshape(-1)

    k = min(int(topk), int(score.numel()))
    if k <= 0:
        return score.sum() * 0.0

    return torch.topk(score, k).values.mean()


# =================== Detection (for early stop + viz) =================== #
def detector_has_detections(yolo, img_rgb: np.ndarray, device, conf: float, iou: float):
    r = yolo.predict(
        cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR),
        conf=conf,
        iou=iou,
        device=device,
        verbose=False,
    )
    return bool(r and r[0].boxes is not None and len(r[0].boxes) > 0)


# =================== XML parser =================== #
def read_bbox_from_xml(xml_path: Path):
    """
    Reads Pascal VOC-style XML and merges all annotated boxes into one enclosing box.
    """
    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    boxes = []
    for obj in root.findall("object"):
        bb = obj.find("bndbox")
        if bb is None:
            continue
        boxes.append([
            float(bb.find("xmin").text),
            float(bb.find("ymin").text),
            float(bb.find("xmax").text),
            float(bb.find("ymax").text),
        ])

    if not boxes:
        raise ValueError(f"No bounding box found in XML: {xml_path}")

    boxes = np.array(boxes, dtype=np.float32)
    return [
        boxes[:, 0].min(),
        boxes[:, 1].min(),
        boxes[:, 2].max(),
        boxes[:, 3].max(),
    ]


def resolve_device(requested_device: str) -> str:
    if requested_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available on this system.")

    return requested_device


# ========================= MAIN ========================= #
def main():
    ap = argparse.ArgumentParser(
        description="Masked baseline I-FGSM attack against YOLOv8 license-plate detection using XML GT boxes."
    )

    ap.add_argument(
        "--weights",
        type=str,
        default="weights/best.pt",
        help="Path to YOLOv8 detector weights.",
    )
    ap.add_argument(
        "--source_dir",
        type=str,
        default="data/images",
        help="Directory containing input images.",
    )
    ap.add_argument(
        "--xml_dir",
        type=str,
        default="data/annotations",
        help="Directory containing Pascal VOC-style XML annotations.",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="outputs/adv_ifgsm",
        help="Directory where outputs will be saved.",
    )

    ap.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Letterbox image size used for the detector.",
    )
    ap.add_argument(
        "--eps",
        type=float,
        default=8 / 255,
        help="Maximum Linf perturbation budget in [0,1] scale.",
    )
    ap.add_argument(
        "--steps",
        type=int,
        default=500,
        help="Number of I-FGSM optimization steps.",
    )
    ap.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Step size per iteration. If omitted, computed as 2*eps/steps.",
    )

    ap.add_argument(
        "--margin",
        type=int,
        default=10,
        help="Extra margin around GT bbox in letterbox space (pixels).",
    )
    ap.add_argument(
        "--topk",
        type=int,
        default=5,
        help="Top-k raw scores used in the differentiable confidence proxy.",
    )

    ap.add_argument(
        "--conf_det",
        type=float,
        default=0.25,
        help="Confidence threshold used for clean detection visualization.",
    )
    ap.add_argument(
        "--iou_det",
        type=float,
        default=0.5,
        help="IoU threshold used for YOLO prediction/NMS.",
    )
    ap.add_argument(
        "--earlystop_conf",
        type=float,
        default=0.25,
        help="Confidence threshold used for early stopping detection check.",
    )

    ap.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Computation device: auto, cpu, cuda, cuda:0, etc.",
    )

    ap.add_argument(
        "--save_plate_patch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to save cropped clean/adversarial plate patches.",
    )
    ap.add_argument(
        "--plate_patch_margin",
        type=int,
        default=10,
        help="Extra crop margin for saved plate patches at original resolution.",
    )
    ap.add_argument(
        "--plate_out_w",
        type=int,
        default=4094,
        help="Optional output width for resized printable adversarial patch. Set <=0 to disable.",
    )
    ap.add_argument(
        "--plate_out_h",
        type=int,
        default=866,
        help="Optional output height for resized printable adversarial patch. Set <=0 to disable.",
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )
    ap.add_argument(
        "--log_every",
        type=int,
        default=5,
        help="Print optimization logs every N iterations.",
    )

    ap.add_argument(
        "--gt_mask_margin",
        type=int,
        default=0,
        help="Margin in original-resolution space for saved overlays/masks.",
    )

    ap.add_argument(
        "--save_adv_original_dir",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to save all adversarial original-resolution images into a flat output folder.",
    )

    args = ap.parse_args()
    seed_all(args.seed)

    validate_inputs(args.weights, args.source_dir, args.xml_dir)

    device = resolve_device(args.device)
    ul_device = 0 if ("cuda" in device and torch.cuda.is_available()) else "cpu"

    model = YOLO(args.weights)
    net = model.model.to(device).eval()
    for p in net.parameters():
        p.requires_grad_(False)

    alpha = args.alpha if args.alpha is not None else 2.0 * (args.eps / max(args.steps, 1))

    ensure_dir(args.out_dir)

    adv_originals_dir = os.path.join(args.out_dir, "adv_originals")
    if args.save_adv_original_dir:
        ensure_dir(adv_originals_dir)

    csv_path = os.path.join(args.out_dir, "psnr_masked.csv")
    csv_rows = [("filename", "psnr_masked")]

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    img_files = [
        p for p in Path(args.source_dir).iterdir()
        if p.is_file() and p.suffix.lower() in exts
    ]
    img_files = sorted(img_files)

    if not img_files:
        raise RuntimeError(f"No supported image files found in: {args.source_dir}")

    for img_path in img_files:
        stem = img_path.stem
        xml_path = Path(args.xml_dir) / f"{stem}.xml"
        if not xml_path.exists():
            print(f"[!] Missing XML for image: {stem}")
            continue

        print(f"\n[→] Processing: {stem}")
        per_dir = os.path.join(args.out_dir, stem)
        ensure_dir(per_dir)

        try:
            bbox = read_bbox_from_xml(xml_path)
            x0, meta, im_rgb, _ = letterbox_to_tensor(str(img_path), args.imgsz, device)
        except Exception as e:
            print(f"[!] Skipping {stem} due to error: {e}")
            continue

        bbox_l = [
            bbox[0] * meta["r"] + meta["left"],
            bbox[1] * meta["r"] + meta["top"],
            bbox[2] * meta["r"] + meta["left"],
            bbox[3] * meta["r"] + meta["top"],
        ]

        bbox_mask = make_bbox_mask(args.imgsz, bbox_l, args.margin, device)
        mask3 = bbox_mask.repeat(1, 3, 1, 1)

        r0 = model.predict(
            str(img_path),
            conf=args.conf_det,
            iou=args.iou_det,
            device=ul_device,
            verbose=False,
        )
        if r0:
            clean_annot = r0[0].plot()
            cv2.imwrite(os.path.join(per_dir, "clean_detection.png"), clean_annot)

        delta = torch.zeros_like(x0, device=device).requires_grad_(True)
        best_adv, early_stopped = None, False

        for t in range(args.steps):
            adv = (x0 + delta * mask3).clamp(0, 1)

            raw = net(adv)
            loss = confidence_proxy(raw, args.topk)

            if (t % max(args.log_every, 1) == 0) or (t == args.steps - 1):
                print(f"  iter {t:03d} | confidence_proxy={loss.item():.6f}")

            adv_orig_rgb = reconstruct_orig_from_letterboxed_rgb(tensor_to_rgb_uint8(adv), meta)
            if not detector_has_detections(model, adv_orig_rgb, ul_device, args.earlystop_conf, args.iou_det):
                print(f"  [✓] Early stop at iter {t}")
                early_stopped = True
                best_adv = adv.detach()
                break

            net.zero_grad(set_to_none=True)
            if delta.grad is not None:
                delta.grad.zero_()

            loss.backward()

            with torch.no_grad():
                # Minimize detector confidence proxy under Linf budget
                delta = (delta - alpha * delta.grad.sign()).clamp(-args.eps, args.eps)
                delta = (delta * mask3).detach().requires_grad_(True)

            best_adv = adv.detach()

        if best_adv is None:
            best_adv = (x0 + delta * mask3).clamp(0, 1).detach()

        psnr_val = float(psnr_masked(x0.detach(), best_adv, bbox_mask).detach().cpu().item())
        clean_letter = tensor_to_rgb_uint8(x0)
        adv_letter = tensor_to_rgb_uint8(best_adv)

        adv_orig = reconstruct_orig_from_letterboxed_rgb(adv_letter, meta)
        clean_orig = im_rgb

        save_rgb(os.path.join(per_dir, "clean_letterboxed.png"), clean_letter)
        save_rgb(os.path.join(per_dir, "adv_letterboxed.png"), adv_letter)
        save_rgb(os.path.join(per_dir, "clean_original.png"), clean_orig)
        save_rgb(os.path.join(per_dir, "adv_original.png"), adv_orig)

        if args.save_adv_original_dir:
            save_rgb(os.path.join(adv_originals_dir, f"{stem}.png"), adv_orig)

        save_gt_masked_area_outputs(
            per_dir=per_dir,
            im_rgb=clean_orig,
            adv_rgb=adv_orig,
            bbox_xyxy=bbox,
            margin=args.gt_mask_margin,
        )

        csv_rows.append((img_path.name, f"{psnr_val:.6f}"))

        if args.save_plate_patch:
            adv_patch, adv_xyxy = crop_with_margin(adv_orig, bbox, args.plate_patch_margin)
            clean_patch, clean_xyxy = crop_with_margin(clean_orig, bbox, args.plate_patch_margin)

            save_rgb(os.path.join(per_dir, "plate_patch_adv.png"), adv_patch)
            save_rgb(os.path.join(per_dir, "plate_patch_clean.png"), clean_patch)

            if args.plate_out_w > 0 and args.plate_out_h > 0:
                adv_res = cv2.resize(
                    adv_patch,
                    (args.plate_out_w, args.plate_out_h),
                    interpolation=cv2.INTER_CUBIC,
                )
                save_rgb(os.path.join(per_dir, "plate_patch_adv_resized.png"), adv_res)

            with open(os.path.join(per_dir, "plate_patch_info.txt"), "w", encoding="utf-8") as f:
                f.write(f"image: {img_path.name}\n")
                f.write(f"xml_bbox_xyxy: {bbox}\n")
                f.write(f"patch_margin_px: {args.plate_patch_margin}\n")
                f.write(f"adv_crop_xyxy: {adv_xyxy}\n")
                f.write(f"clean_crop_xyxy: {clean_xyxy}\n")
                f.write(f"orig_shape: {clean_orig.shape}\n")
                if args.plate_out_w > 0 and args.plate_out_h > 0:
                    f.write(f"resized_to: {(args.plate_out_w, args.plate_out_h)}\n")

        with open(os.path.join(per_dir, "summary.txt"), "w", encoding="utf-8") as f:
            f.write(f"image: {img_path.name}\n")
            f.write(f"steps: {args.steps}, eps: {args.eps}, alpha: {alpha}\n")
            f.write(f"topk: {args.topk}\n")
            f.write(f"conf_det: {args.conf_det}, iou_det: {args.iou_det}\n")
            f.write(f"earlystop_conf: {args.earlystop_conf}\n")
            f.write(f"seed: {args.seed}\n")
            f.write(f"early_stopped: {early_stopped}\n")
            f.write(f"psnr_masked: {psnr_val:.6f}\n")
            f.write(f"mask_margin_letterbox_px: {args.margin}\n")
            f.write(f"gt_mask_margin_orig_px: {args.gt_mask_margin}\n")
            f.write(f"save_adv_original_dir: {args.save_adv_original_dir}\n")

        print(f"  [✓] Saved outputs to: {per_dir}")
        print(f"  [✓] PSNR(masked): {psnr_val:.3f} dB")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(csv_rows)

    print("\n✓ Done.")
    print(f"CSV saved to: {csv_path}")
    if args.save_adv_original_dir:
        print(f"All adversarial originals saved in: {adv_originals_dir}")


if __name__ == "__main__":
    main()
