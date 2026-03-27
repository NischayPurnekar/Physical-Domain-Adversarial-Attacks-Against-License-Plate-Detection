#!/usr/bin/env python3
"""
Masked I-FGSM adversarial attack for YOLOv8 license-plate detection
using ground-truth Pascal VOC XML bounding boxes, with EOT-based optimization
and post-attack robustness evaluation.

Overview
--------
This script:
- reads GT bounding boxes from XML annotations
- letterboxes images to the detector input size while preserving resize/padding metadata
- perturbs only the GT license-plate region in letterbox space
- minimizes a score-aligned differentiable proxy derived from YOLO raw outputs
- applies Expectation Over Transformation (EOT) during optimization
- optionally saves transformed samples during optimization
- performs post-attack robustness evaluation under the same EOT distribution
- saves per-image artifacts, summaries, and CSV files

Important compatibility note
----------------------------
This script assumes that the raw forward pass of the loaded YOLOv8 model returns
a tuple in which `raw[1]["scores"]` exists and has shape compatible with `[B, 1, N]`.
This behavior depends on the specific Ultralytics version used in the experiments.
Please pin the Ultralytics version in your environment for reproducibility.

Runtime note
------------
This script can be computationally expensive because:
- each optimization step averages loss over multiple EOT transforms
- optional robustness evaluation uses additional transformed samples
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
import torch.nn.functional as F
import torchvision.transforms.functional as TF


# ========================= Reproducibility ========================= #
def seed_all(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
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


def resolve_device(requested_device: str) -> str:
    if requested_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available on this system.")

    return requested_device


# ========================= EOT ========================= #
class EOTWrapper:
    """
    Differentiable EOT wrapper for physical-like variations.
    Input:  [1, 3, H, W]
    Output: [N, 3, H, W] in [0, 1]
    """

    def __init__(
        self,
        n=10,
        p_flip=0.0,
        degrees=15.0,
        translate=0.08,
        d_ref=0.5,
        d_min=0.5,
        d_max=2.5,
        perspective=0.004,
        shear=4.0,
        p_blur=0.5,
        motion_blur_max=9,
        gaussian_blur_sigma=(0.3, 1.6),
        p_illum=0.6,
        brightness=(0.75, 1.25),
        contrast=(0.75, 1.25),
        gamma=(0.85, 1.20),
        p_resample=0.35,
        resample_range=(0.55, 1.00),
        fill=(114 / 255.0,),
    ):
        self.n = int(n)
        self.p_flip = float(p_flip)
        self.degrees = float(degrees)
        self.translate = float(translate)

        self.d_ref = float(d_ref)
        self.d_min = float(d_min)
        self.d_max = float(d_max)
        if not (self.d_min > 0 and self.d_max >= self.d_min and self.d_ref > 0):
            raise ValueError("Invalid distance parameters: require d_min>0, d_max>=d_min, d_ref>0")

        self.perspective = float(perspective)
        self.shear = float(shear)
        self.fill = fill

        self.p_blur = float(p_blur)
        self.motion_blur_max = int(motion_blur_max)
        self.gaussian_blur_sigma = tuple(map(float, gaussian_blur_sigma))

        self.p_illum = float(p_illum)
        self.brightness = brightness
        self.contrast = contrast
        self.gamma = gamma

        self.p_resample = float(p_resample)
        self.resample_range = tuple(map(float, resample_range))

    def _rand_uniform(self, device, a, b):
        return float(torch.empty((), device=device).uniform_(a, b).item())

    def _apply_motion_blur(self, img_chw: torch.Tensor, ksize: int) -> torch.Tensor:
        if ksize <= 1:
            return img_chw

        if ksize % 2 == 0:
            ksize += 1

        kernel = torch.zeros((1, 1, ksize, ksize), device=img_chw.device)
        angle = self._rand_uniform(img_chw.device, 0, 180)
        c = ksize // 2

        if 45 < angle < 135:
            kernel[:, :, :, c] = 1.0
        else:
            kernel[:, :, c, :] = 1.0

        kernel = kernel / kernel.sum().clamp_min(1e-12)
        y = F.conv2d(
            img_chw.unsqueeze(0),
            kernel.repeat(3, 1, 1, 1),
            padding=ksize // 2,
            groups=3,
        )
        return y.squeeze(0)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        if not (x.ndim == 4 and x.shape[0] == 1):
            raise ValueError("EOT expects input of shape [1, 3, H, W]")

        device = x.device
        _, _, H, W = x.shape
        outs = []

        for _ in range(self.n):
            img = x[0]

            if torch.rand((), device=device) < self.p_flip:
                img = TF.hflip(img)

            d = self._rand_uniform(device, self.d_min, self.d_max)
            scale = self.d_ref / d

            angle = self._rand_uniform(device, -self.degrees, self.degrees)
            sh = self._rand_uniform(device, -self.shear, self.shear)

            max_dx = int(W * self.translate)
            max_dy = int(H * self.translate)
            tx = int(self._rand_uniform(device, -max_dx, max_dx))
            ty = int(self._rand_uniform(device, -max_dy, max_dy))

            img = TF.affine(
                img,
                angle=angle,
                translate=[tx, ty],
                scale=scale,
                shear=[sh, 0.0],
                interpolation=TF.InterpolationMode.BILINEAR,
                fill=self.fill,
            )

            if self.perspective > 0:
                sp = torch.tensor(
                    [[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]],
                    dtype=torch.float32,
                    device=device,
                )
                ep = sp.clone()
                ep[:, 0] += torch.empty((4,), device=device).uniform_(-self.perspective * W, self.perspective * W)
                ep[:, 1] += torch.empty((4,), device=device).uniform_(-self.perspective * H, self.perspective * H)
                img = TF.perspective(
                    img,
                    sp.tolist(),
                    ep.tolist(),
                    interpolation=TF.InterpolationMode.BILINEAR,
                    fill=self.fill,
                )

            if torch.rand((), device=device) < self.p_resample:
                s = self._rand_uniform(device, self.resample_range[0], self.resample_range[1])
                h2 = max(2, int(round(H * s)))
                w2 = max(2, int(round(W * s)))
                img = TF.resize(img, size=[h2, w2], antialias=True)
                img = TF.resize(img, size=[H, W], antialias=True)

            if torch.rand((), device=device) < self.p_blur:
                if torch.rand((), device=device) < 0.5:
                    sigma = self._rand_uniform(device, *self.gaussian_blur_sigma)
                    k = int(round(3 * sigma)) * 2 + 1
                    img = TF.gaussian_blur(img, [k, k], sigma=(sigma, sigma))
                else:
                    k = int(self._rand_uniform(device, 3, max(3, self.motion_blur_max)))
                    img = self._apply_motion_blur(img, k)

            if torch.rand((), device=device) < self.p_illum:
                b = self._rand_uniform(device, *self.brightness)
                c = self._rand_uniform(device, *self.contrast)
                g = self._rand_uniform(device, *self.gamma)
                img = TF.adjust_brightness(img, b)
                img = TF.adjust_contrast(img, c)
                img = TF.adjust_gamma(img, g)

            img = TF.resize(img, size=[H, W], antialias=True)
            outs.append(img.clamp(0, 1))

        return torch.stack(outs, dim=0)


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


# =================== Score-aligned loss =================== #
def topk_score_loss_from_raw(raw, topk: int):
    """
    Assumes raw is a tuple such that:
      raw[1]["scores"] has shape [B, 1, N] or is reshape-compatible.

    The attack minimizes mean(top-k scores) to suppress detections.
    """
    if not (isinstance(raw, tuple) and len(raw) >= 2 and isinstance(raw[1], dict) and "scores" in raw[1]):
        raise RuntimeError(
            "Unexpected YOLO raw output format. "
            "This script expects raw[1]['scores'] from the pinned Ultralytics version."
        )

    scores = raw[1]["scores"]
    if scores.ndim != 3:
        scores = scores.reshape(scores.shape[0], 1, -1)

    s = scores[0, 0]
    k = min(int(topk), int(s.numel()))
    if k <= 0:
        return s.sum() * 0.0, s

    loss = torch.topk(s, k).values.mean()
    return loss, s


# =================== Detection helpers =================== #
def detector_predict(yolo, img_rgb, device, conf, iou):
    return yolo.predict(
        cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR),
        conf=conf,
        iou=iou,
        device=device,
        verbose=False,
    )


def detector_has_detections(yolo, img_rgb, device, conf, iou):
    r = detector_predict(yolo, img_rgb, device, conf, iou)
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


# =================== Optional heatmap debug =================== #
def scores_to_heatmap_640(scores_8400: torch.Tensor, imgsz: int = 640):
    """
    Converts YOLOv8 candidate scores at imgsz=640 into a coarse heatmap.
    Assumes the standard 80x80, 40x40, 20x20 candidate decomposition (total 8400).
    """
    if imgsz != 640:
        raise ValueError("Heatmap mapper is implemented for imgsz=640 only.")

    s = scores_8400.detach().float()
    s8 = s[:6400].reshape(80, 80)
    s16 = s[6400:6400 + 1600].reshape(40, 40)
    s32 = s[6400 + 1600:].reshape(20, 20)

    s16u = torch.nn.functional.interpolate(s16[None, None], size=(80, 80), mode="bilinear", align_corners=False)[0, 0]
    s32u = torch.nn.functional.interpolate(s32[None, None], size=(80, 80), mode="bilinear", align_corners=False)[0, 0]
    s80 = torch.max(torch.max(s8, s16u), s32u)

    s640 = torch.nn.functional.interpolate(s80[None, None], size=(640, 640), mode="bilinear", align_corners=False)[0, 0]
    return s640.cpu().numpy()


def save_heatmap(path: str, hm: np.ndarray):
    h = hm - hm.min()
    if h.max() > 1e-9:
        h = h / h.max()
    img = (h * 255).astype(np.uint8)
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    cv2.imwrite(path, img)


# =================== Robustness evaluation under EOT =================== #
@torch.no_grad()
def eval_robustness_under_eot(
    model_yolo: YOLO,
    eot: EOTWrapper,
    best_adv_letterboxed: torch.Tensor,
    meta: dict,
    ul_device,
    conf: float,
    iou: float,
    save_dir: str = None,
    save_max: int = 10,
):
    """
    Returns:
      detection_rate: fraction of EOT transforms that still have detections
      success_rate:   fraction with no detections
    """
    if not (best_adv_letterboxed.ndim == 4 and best_adv_letterboxed.shape[0] == 1):
        raise ValueError("best_adv_letterboxed must have shape [1, 3, H, W]")

    adv_eot = eot.apply(best_adv_letterboxed)
    n = int(adv_eot.shape[0])

    det_flags = []
    saved = 0

    if save_dir is not None:
        ensure_dir(save_dir)

    for i in range(n):
        letter_i = tensor_to_rgb_uint8(adv_eot[i:i + 1])
        orig_i = reconstruct_orig_from_letterboxed_rgb(letter_i, meta)

        has_det = detector_has_detections(model_yolo, orig_i, ul_device, conf, iou)
        det_flags.append(1 if has_det else 0)

        if save_dir is not None and saved < int(save_max):
            save_rgb(os.path.join(save_dir, f"robust_eot_{i:03d}_orig.png"), orig_i)

            r = detector_predict(model_yolo, orig_i, ul_device, conf, iou)
            if r:
                det_plot = r[0].plot()
                cv2.imwrite(os.path.join(save_dir, f"robust_eot_{i:03d}_det.png"), det_plot)

            saved += 1

    det_rate = float(np.mean(det_flags)) if det_flags else 0.0
    success_rate = 1.0 - det_rate
    return det_rate, success_rate


# ========================= MAIN ========================= #
def main():
    ap = argparse.ArgumentParser(
        description="Masked I-FGSM adversarial attack with EOT and robustness evaluation for YOLOv8 license-plate detection."
    )

    ap.add_argument("--weights", type=str, default="weights/best.pt", help="Path to YOLOv8 detector weights.")
    ap.add_argument("--source_dir", type=str, default="data/images", help="Directory containing input images.")
    ap.add_argument("--xml_dir", type=str, default="data/annotations", help="Directory containing Pascal VOC-style XML annotations.")
    ap.add_argument("--out_dir", type=str, default="outputs/adv_ifgsm_eot", help="Directory where outputs will be saved.")

    ap.add_argument("--imgsz", type=int, default=640, help="Letterbox image size used for the detector.")
    ap.add_argument("--eps", type=float, default=48 / 255, help="Maximum Linf perturbation budget in [0,1] scale.")
    ap.add_argument("--steps", type=int, default=1500, help="Number of attack optimization steps.")
    ap.add_argument("--alpha", type=float, default=8 / 255, help="Step size per iteration. Set explicitly to match the experiment.")

    ap.add_argument("--margin", type=int, default=10, help="Extra margin around GT bbox in letterbox space (pixels).")
    ap.add_argument("--topk", type=int, default=200, help="Top-k raw candidate scores used in the differentiable suppression loss.")

    ap.add_argument("--conf_det", type=float, default=0.25, help="Confidence threshold used for clean detection visualization.")
    ap.add_argument("--iou_det", type=float, default=0.5, help="IoU threshold used for YOLO prediction/NMS.")
    ap.add_argument("--earlystop_conf", type=float, default=0.25, help="Confidence threshold used for early stopping on the untransformed adversarial image.")

    ap.add_argument("--device", type=str, default="auto", help="Computation device: auto, cpu, cuda, cuda:0, etc.")

    ap.add_argument("--save_plate_patch", action=argparse.BooleanOptionalAction, default=False, help="Whether to save cropped clean/adversarial plate patches.")
    ap.add_argument("--plate_patch_margin", type=int, default=10, help="Extra crop margin for saved plate patches at original resolution.")
    ap.add_argument("--plate_out_w", type=int, default=4094, help="Optional output width for resized printable adversarial patch. Set <=0 to disable.")
    ap.add_argument("--plate_out_h", type=int, default=866, help="Optional output height for resized printable adversarial patch. Set <=0 to disable.")

    ap.add_argument("--seed", type=int, default=0, help="Random seed.")
    ap.add_argument("--log_every", type=int, default=1, help="Print optimization logs every N iterations.")
    ap.add_argument("--gt_mask_margin", type=int, default=0, help="Margin in original-resolution space for saved overlays/masks.")
    ap.add_argument("--save_adv_original_dir", action=argparse.BooleanOptionalAction, default=True, help="Whether to save all adversarial original-resolution images into a flat output folder.")

    ap.add_argument("--debug_heatmap", action=argparse.BooleanOptionalAction, default=False, help="Whether to save optional score heatmaps during optimization.")
    ap.add_argument("--debug_heatmap_every", type=int, default=25, help="Save a debug heatmap every N iterations.")

    # EOT args
    ap.add_argument("--eot_n", type=int, default=10, help="Number of EOT transforms per optimization step.")
    ap.add_argument("--eot_p_flip", type=float, default=0.0, help="Probability of horizontal flip.")
    ap.add_argument("--eot_degrees", type=float, default=6.0, help="Maximum absolute rotation angle in degrees.")
    ap.add_argument("--eot_translate", type=float, default=0.02, help="Maximum translation fraction relative to image size.")
    ap.add_argument("--eot_d_ref", type=float, default=0.9, help="Reference distance parameter used to derive scale.")
    ap.add_argument("--eot_d_min", type=float, default=0.85, help="Minimum sampled distance parameter.")
    ap.add_argument("--eot_d_max", type=float, default=1.15, help="Maximum sampled distance parameter.")
    ap.add_argument("--eot_perspective", type=float, default=0.0008, help="Perspective distortion magnitude.")
    ap.add_argument("--eot_shear", type=float, default=1.5, help="Maximum shear angle.")
    ap.add_argument("--eot_p_blur", type=float, default=0.1, help="Probability of blur.")
    ap.add_argument("--eot_motion_blur_max", type=int, default=9, help="Maximum motion-blur kernel size.")
    ap.add_argument("--eot_gauss_sigma_min", type=float, default=0.3, help="Minimum Gaussian blur sigma.")
    ap.add_argument("--eot_gauss_sigma_max", type=float, default=1.6, help="Maximum Gaussian blur sigma.")
    ap.add_argument("--eot_p_illum", type=float, default=0.20, help="Probability of illumination perturbation.")
    ap.add_argument("--eot_brightness_min", type=float, default=0.75, help="Minimum brightness factor.")
    ap.add_argument("--eot_brightness_max", type=float, default=1.25, help="Maximum brightness factor.")
    ap.add_argument("--eot_contrast_min", type=float, default=0.75, help="Minimum contrast factor.")
    ap.add_argument("--eot_contrast_max", type=float, default=1.25, help="Maximum contrast factor.")
    ap.add_argument("--eot_gamma_min", type=float, default=0.85, help="Minimum gamma factor.")
    ap.add_argument("--eot_gamma_max", type=float, default=1.20, help="Maximum gamma factor.")
    ap.add_argument("--eot_p_resample", type=float, default=0.05, help="Probability of down/up resampling.")
    ap.add_argument("--eot_resample_min", type=float, default=0.40, help="Minimum resampling scale factor.")
    ap.add_argument("--eot_resample_max", type=float, default=1.00, help="Maximum resampling scale factor.")

    ap.add_argument("--save_eot_samples", action=argparse.BooleanOptionalAction, default=False, help="Whether to save some EOT-transformed samples during optimization.")
    ap.add_argument("--save_eot_every", type=int, default=25, help="Save EOT samples every N iterations.")
    ap.add_argument("--eot_save_max_per_iter", type=int, default=8, help="Maximum number of EOT samples to save per selected iteration.")

    ap.add_argument("--robust_eval", action=argparse.BooleanOptionalAction, default=True, help="Whether to evaluate final robustness under the same EOT distribution.")
    ap.add_argument("--robust_n", type=int, default=50, help="Number of EOT samples used for robustness evaluation.")
    ap.add_argument("--robust_conf", type=float, default=0.25, help="Confidence threshold used during robustness evaluation.")
    ap.add_argument("--robust_iou", type=float, default=0.5, help="IoU threshold used during robustness evaluation.")
    ap.add_argument("--save_robust_samples", action=argparse.BooleanOptionalAction, default=False, help="Whether to save some transformed robustness samples and detection plots.")
    ap.add_argument("--robust_save_max", type=int, default=10, help="Maximum number of robustness samples to save per image.")

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

    eot = EOTWrapper(
        n=args.eot_n,
        p_flip=args.eot_p_flip,
        degrees=args.eot_degrees,
        translate=args.eot_translate,
        d_ref=args.eot_d_ref,
        d_min=args.eot_d_min,
        d_max=args.eot_d_max,
        perspective=args.eot_perspective,
        shear=args.eot_shear,
        p_blur=args.eot_p_blur,
        motion_blur_max=args.eot_motion_blur_max,
        gaussian_blur_sigma=(args.eot_gauss_sigma_min, args.eot_gauss_sigma_max),
        p_illum=args.eot_p_illum,
        brightness=(args.eot_brightness_min, args.eot_brightness_max),
        contrast=(args.eot_contrast_min, args.eot_contrast_max),
        gamma=(args.eot_gamma_min, args.eot_gamma_max),
        p_resample=args.eot_p_resample,
        resample_range=(args.eot_resample_min, args.eot_resample_max),
    )

    eot_robust = EOTWrapper(
        n=args.robust_n,
        p_flip=args.eot_p_flip,
        degrees=args.eot_degrees,
        translate=args.eot_translate,
        d_ref=args.eot_d_ref,
        d_min=args.eot_d_min,
        d_max=args.eot_d_max,
        perspective=args.eot_perspective,
        shear=args.eot_shear,
        p_blur=args.eot_p_blur,
        motion_blur_max=args.eot_motion_blur_max,
        gaussian_blur_sigma=(args.eot_gauss_sigma_min, args.eot_gauss_sigma_max),
        p_illum=args.eot_p_illum,
        brightness=(args.eot_brightness_min, args.eot_brightness_max),
        contrast=(args.eot_contrast_min, args.eot_contrast_max),
        gamma=(args.eot_gamma_min, args.eot_gamma_max),
        p_resample=args.eot_p_resample,
        resample_range=(args.eot_resample_min, args.eot_resample_max),
    )

    ensure_dir(args.out_dir)

    adv_originals_dir = os.path.join(args.out_dir, "adv_originals")
    if args.save_adv_original_dir:
        ensure_dir(adv_originals_dir)

    eot_samples_root = os.path.join(args.out_dir, "eot_samples_during_attack")
    if args.save_eot_samples:
        ensure_dir(eot_samples_root)

    robust_root = os.path.join(args.out_dir, "robustness_eot")
    if args.robust_eval and args.save_robust_samples:
        ensure_dir(robust_root)

    psnr_csv = os.path.join(args.out_dir, "psnr_masked.csv")
    robust_csv = os.path.join(args.out_dir, "robustness_eot.csv")

    psnr_rows = [("filename", "psnr_masked")]
    robust_rows = [("filename", "det_rate_eot", "robust_success_eot", "robust_n", "robust_conf", "robust_iou")]

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    img_files = sorted([p for p in Path(args.source_dir).iterdir() if p.is_file() and p.suffix.lower() in exts])

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
            print(f"[!] Skipping {stem} due to input error: {e}")
            continue

        bbox_l = [
            bbox[0] * meta["r"] + meta["left"],
            bbox[1] * meta["r"] + meta["top"],
            bbox[2] * meta["r"] + meta["left"],
            bbox[3] * meta["r"] + meta["top"],
        ]

        bbox_mask = make_bbox_mask(args.imgsz, bbox_l, args.margin, device)
        mask3 = bbox_mask.repeat(1, 3, 1, 1)

        r0 = model.predict(str(img_path), conf=args.conf_det, iou=args.iou_det, device=ul_device, verbose=False)
        if r0:
            clean_annot = r0[0].plot()
            cv2.imwrite(os.path.join(per_dir, "clean_detection.png"), clean_annot)

        delta = torch.zeros_like(x0, device=device).requires_grad_(True)
        best_adv, early_stopped = None, False

        try:
            for t in range(args.steps):
                adv = (x0 + delta * mask3).clamp(0, 1)
                adv_eot = eot.apply(adv)

                if args.save_eot_samples and args.save_eot_every > 0 and (t % args.save_eot_every == 0):
                    eot_dir = os.path.join(eot_samples_root, stem, f"iter_{t:04d}")
                    ensure_dir(eot_dir)
                    ksave = min(int(args.eot_save_max_per_iter), int(adv_eot.shape[0]))
                    for i in range(ksave):
                        save_rgb(os.path.join(eot_dir, f"eot_{i:02d}_letter.png"), tensor_to_rgb_uint8(adv_eot[i:i + 1]))
                        orig_i = reconstruct_orig_from_letterboxed_rgb(tensor_to_rgb_uint8(adv_eot[i:i + 1]), meta)
                        save_rgb(os.path.join(eot_dir, f"eot_{i:02d}_orig.png"), orig_i)

                per_losses = []
                scores_for_debug = None

                for i in range(int(adv_eot.shape[0])):
                    raw_i = net(adv_eot[i:i + 1])
                    li, si = topk_score_loss_from_raw(raw_i, args.topk)
                    per_losses.append(li)
                    if scores_for_debug is None:
                        scores_for_debug = si

                loss = torch.stack(per_losses).mean()

                mv, mi = scores_for_debug.max(dim=0)
                mv_prob = mv.sigmoid()

                if (t % max(args.log_every, 1) == 0) or (t == args.steps - 1):
                    L = torch.stack(per_losses)
                    lmin = float(L.min().detach().cpu().item())
                    lmax = float(L.max().detach().cpu().item())
                    print(
                        f"  iter {t:03d} | EOT loss(topk_mean)={loss.item():.6f} | "
                        f"min={lmin:.6f} max={lmax:.6f} | "
                        f"max_logit={mv.item():.3f} prob={mv_prob.item():.4f} idx={int(mi.item())}"
                    )

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

                if (t % max(args.log_every, 1) == 0) or (t == args.steps - 1):
                    with torch.no_grad():
                        g = delta.grad
                        gm = g.abs().mean().item()
                        gmx = g.abs().max().item()
                        nz = (g.abs() > 0).float().mean().item() * 100.0
                        sat = (delta.abs() >= (args.eps - 1e-12)).float().mean().item() * 100.0
                        dinf = delta.abs().max().item()
                        print(f"    grad | mean={gm:.3e} max={gmx:.3e} nonzero%={nz:.2f}")
                        print(f"    delt | Linf={dinf:.5f} sat%={sat:.2f}")

                if args.debug_heatmap and args.imgsz == 640 and (t % max(args.debug_heatmap_every, 1) == 0):
                    hm = scores_to_heatmap_640(scores_for_debug, imgsz=640)
                    save_heatmap(os.path.join(per_dir, f"score_heatmap_iter_{t:04d}.png"), hm)

                with torch.no_grad():
                    delta = (delta - alpha * delta.grad.sign()).clamp(-args.eps, args.eps)
                    delta = (delta * mask3).detach().requires_grad_(True)

                best_adv = adv.detach()

        except Exception as e:
            print(f"[!] Attack failed for {stem}: {e}")
            continue

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

        psnr_rows.append((img_path.name, f"{psnr_val:.6f}"))

        if args.save_plate_patch:
            adv_patch, adv_xyxy = crop_with_margin(adv_orig, bbox, args.plate_patch_margin)
            clean_patch, clean_xyxy = crop_with_margin(clean_orig, bbox, args.plate_patch_margin)

            save_rgb(os.path.join(per_dir, "plate_patch_adv.png"), adv_patch)
            save_rgb(os.path.join(per_dir, "plate_patch_clean.png"), clean_patch)

            if args.plate_out_w > 0 and args.plate_out_h > 0:
                adv_res = cv2.resize(adv_patch, (args.plate_out_w, args.plate_out_h), interpolation=cv2.INTER_CUBIC)
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

        det_rate, success_rate = None, None
        if args.robust_eval:
            robust_dir = None
            if args.save_robust_samples:
                robust_dir = os.path.join(robust_root, stem)

            det_rate, success_rate = eval_robustness_under_eot(
                model_yolo=model,
                eot=eot_robust,
                best_adv_letterboxed=best_adv,
                meta=meta,
                ul_device=ul_device,
                conf=args.robust_conf,
                iou=args.robust_iou,
                save_dir=robust_dir,
                save_max=args.robust_save_max,
            )

            robust_rows.append((
                img_path.name,
                f"{det_rate:.6f}",
                f"{success_rate:.6f}",
                str(args.robust_n),
                str(args.robust_conf),
                str(args.robust_iou),
            ))

            print(f"  [ROBUST] EOT det_rate={det_rate:.3f} | robust_success(no-det)={success_rate:.3f} (N={args.robust_n})")

        with open(os.path.join(per_dir, "summary.txt"), "w", encoding="utf-8") as f:
            f.write(f"image: {img_path.name}\n")
            f.write(f"steps: {args.steps}, eps: {args.eps}, alpha: {alpha}\n")
            f.write(f"topk(scores): {args.topk}\n")
            f.write(f"conf_det: {args.conf_det}, iou_det: {args.iou_det}\n")
            f.write(f"earlystop_conf: {args.earlystop_conf}\n")
            f.write(f"seed: {args.seed}\n")
            f.write(f"early_stopped: {early_stopped}\n")
            f.write(f"psnr_masked: {psnr_val:.6f}\n")
            f.write(f"mask_margin_letterbox_px: {args.margin}\n")
            f.write(f"gt_mask_margin_orig_px: {args.gt_mask_margin}\n")
            f.write(f"save_adv_original_dir: {args.save_adv_original_dir}\n")
            f.write(f"debug_heatmap: {args.debug_heatmap}, every={args.debug_heatmap_every}\n")

            f.write("\n[EOT]\n")
            f.write(f"eot_n(opt): {args.eot_n}\n")
            f.write(f"degrees: {args.eot_degrees}\n")
            f.write(f"translate: {args.eot_translate}\n")
            f.write(f"d_ref/d_min/d_max: {args.eot_d_ref}/{args.eot_d_min}/{args.eot_d_max}\n")
            f.write(f"perspective: {args.eot_perspective}\n")
            f.write(f"shear: {args.eot_shear}\n")
            f.write(f"p_blur: {args.eot_p_blur}\n")
            f.write(f"p_illum: {args.eot_p_illum}\n")
            f.write(f"p_resample: {args.eot_p_resample}\n")
            f.write(f"save_eot_samples(during): {args.save_eot_samples}, every={args.save_eot_every}, max_per_iter={args.eot_save_max_per_iter}\n")

            f.write("\n[ROBUSTNESS_EOT]\n")
            f.write(f"robust_eval: {args.robust_eval}\n")
            f.write(f"robust_n: {args.robust_n}\n")
            f.write(f"robust_conf/iou: {args.robust_conf}/{args.robust_iou}\n")
            if det_rate is not None:
                f.write(f"det_rate_eot: {det_rate:.6f}\n")
                f.write(f"robust_success_eot: {success_rate:.6f}\n")
            f.write(f"save_robust_samples: {args.save_robust_samples}, robust_save_max: {args.robust_save_max}\n")

        print(f"  [✓] Saved outputs to: {per_dir}")
        print(f"  [✓] PSNR(masked): {psnr_val:.3f} dB")

    with open(psnr_csv, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(psnr_rows)

    with open(robust_csv, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(robust_rows)

    print("\n✓ Done.")
    print(f"PSNR CSV saved to: {psnr_csv}")
    print(f"Robustness CSV saved to: {robust_csv}")
    if args.save_adv_original_dir:
        print(f"All adversarial originals saved in: {adv_originals_dir}")
    if args.save_eot_samples:
        print(f"EOT samples during attack saved in: {eot_samples_root}")
    if args.robust_eval and args.save_robust_samples:
        print(f"Robustness transformed samples saved in: {robust_root}")


if __name__ == "__main__":
    main()
