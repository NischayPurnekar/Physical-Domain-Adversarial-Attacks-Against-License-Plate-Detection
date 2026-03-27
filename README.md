# Physical-Domain Adversarial Attacks Against License-Plate Detection

This repository contains the **experimental code** for physical-domain adversarial attacks against license-plate detection (LPD), including baseline masked attacks, EOT-based attacks, optional print-scan simulation, and Lab-L constrained attack variants.

## Repository scope

This repository provides the **code for the experiments only**.

The **dataset and corresponding annotations** used for generating physical adversarial license plates are available in a separate repository:

**Dataset repository:** `https://github.com/NischayPurnekar/license-plate-dataset`

The **print-and-scan simulator** used in the print-scan-enabled attack variants is available in a separate repository:

**Simulator repository:** `https://github.com/NischayPurnekar/print-and-scan-simulator`

## Included scripts

- `attack_baseline_ifgsm.py`  
  Baseline masked I-FGSM attack using ground-truth XML bounding boxes.

- `attack_physical_eot.py`  
  EOT-based physical attack against YOLOv8 license-plate detection.

- `attack_physical_eot_ps.py`  
  EOT-based physical attack with optional differentiable print-scan simulation.

- `attack_lab_l_char_mask_eot.py`  
  Lab-L-only perturbation attack with character exclusion masking and EOT.

- `attack_lab_l_roi_eot_ps.py`  
  Advanced Lab-L-only attack with center-ROI character masking, optional print-scan simulation, epsilon warmup, NaN handling, EOT early stop, and robustness evaluation.

- `inference.py`  
  Inference/evaluation script for running the detector on clean or adversarial images.

- `networks.py`  
  Required helper module for the print-scan-enabled scripts when using the CycleGAN-based simulator.

## Main features

Depending on the selected script, the repository includes support for:

- masked perturbations constrained to the license-plate region
- XML-guided attack masking using Pascal VOC-style annotations
- Expectation Over Transformation (EOT)
- robustness evaluation under the same EOT distribution
- Lab-L-only perturbations
- character exclusion masks
- center-ROI character masking
- optional differentiable print-scan simulation
- luminance perturbation saving (`deltaL`)
- per-image summaries, visualizations, and CSV reports

## Installation

### Option 1: Conda

```bash
conda env create -f environment.yml
conda activate github_attacks_lpd
```

### Option 2: pip

```bash
pip install -r requirements.txt
```

## Requirements and notes

- Python 3.10 is recommended.
- YOLO detector weights are **included** in this repository.
- For scripts using the print-scan simulator, the corresponding CycleGAN generator checkpoint must be provided separately.
- `networks.py` must be present for print-scan-enabled scripts.
- Some scripts assume a specific Ultralytics raw-output structure (YOLOv8), in particular the availability of `raw[1]["scores"]`. For reproducibility, pin the Ultralytics version used in `requirements.txt` or `environment.yml`.

## Expected annotation format

The attack scripts expect **Pascal VOC-style XML annotations** with bounding boxes stored under `object -> bndbox`.

## Example usage

### Baseline masked attack

```bash
python attack_baseline_ifgsm.py \
  --weights /path/to/best.pt \
  --source_dir /path/to/images \
  --xml_dir /path/to/annotations \
  --out_dir /path/to/output \
  --device cuda:0
```

### EOT-based physical attack

```bash
python attack_physical_eot.py \
  --weights /path/to/best.pt \
  --source_dir /path/to/images \
  --xml_dir /path/to/annotations \
  --out_dir /path/to/output \
  --device cuda:0
```

### EOT + print-scan simulation

```bash
python attack_physical_eot_ps.py \
  --weights /path/to/best.pt \
  --source_dir /path/to/images \
  --xml_dir /path/to/annotations \
  --out_dir /path/to/output \
  --ps_model_path /path/to/ps_generator.pth \
  --device cuda:0
```

### Lab-L-only attack with center-ROI character masking

```bash
python attack_lab_l_roi_eot_ps.py \
  --weights /path/to/best.pt \
  --source_dir /path/to/images \
  --xml_dir /path/to/annotations \
  --out_dir /path/to/output \
  --device cuda:0
```

## Outputs

The scripts may generate, depending on the selected configuration:

- adversarial images in letterboxed and original resolution
- clean/adversarial crops and overlays
- detector visualizations
- per-image `summary.txt`
- `psnr_masked.csv`
- `robustness_eot.csv`
- optional saved EOT samples
- optional saved luminance perturbation artifacts
- optional flat folders containing adversarial originals

## What is not included

This repository does **not** include:

- training or test datasets
- print-scan generator checkpoints
- generated attack outputs
- demo or presentation materials

## Contact

For questions related to the experimental code release, please contact:

**nischay.purnekar@student.unisi.it**
