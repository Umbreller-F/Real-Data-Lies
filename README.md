<div align="center">
<br>
<h3>Real Data Lies: Unveiling and Closing the Quality Shortcut in Generalizable AI-Generated Video Detection</h3>

Ziyuan Fang<sup>1</sup>, Tianyi Wei<sup>2†</sup>, Guanjie Wang<sup>1</sup>, Weiming Zhang<sup>1</sup>, Nenghai Yu<sup>1</sup>, Wenbo Zhou<sup>1†</sup>

<div class="is-size-6 publication-authors">
  <p class="footnote">
    <span class="footnote-symbol"><sup>†</sup></span>Corresponding author
  </p>
</div>

<sup>1</sup>University of Science and Technology of China <sup>2</sup>Nanyang Technological University
</div>

---

## Overview

This repository is the official implementation of **"Real Data Lies: Unveiling and Closing the Quality Shortcut in Generalizable AI-Generated Video Detection"**. It contains:

- **RDL benchmark**: a large-scale benchmark for AI-generated video detection, pairing real videos collected from 9 public sources with fake videos produced by 19 generation models, under three real-data quality alignment settings:
  - **Biased**: real videos from Kinetics-400 (quality-mismatched with fakes, the common practice);
  - **Aligned**: real videos from InternVid-AES (quality-aligned with fakes);
  - **Expanded**: the aligned setting plus two quality-extreme training pairs — low-quality (Kinetics-400 real / OpenSora fake) and ultra-high-quality (OpenVidHD real / DynamicCrafter fake).
- **Unified training / evaluation code** for 9+ detectors: TimeSformer (SSV2 / K400 / scratch), VideoMAEv2, DeMamba, DINOv2, DINOv3, NPR, SAFE, AIDE, and NSG-VD.
- **Data analysis code** (`data_analysis/`) for video quality assessment (DOVER), quality-score-based sampling, and training-data distribution visualization.

## Installation

```bash
conda create -n RDL python=3.10 -y
conda activate RDL

# Install PyTorch (CUDA 12.8 build used by the authors)
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128

# Install the remaining dependencies
pip install -r requirements.txt
```

Notes:

- `requirements.txt` was exported from the authors' environment and pruned to the packages actually imported by this project. `clip` is installed from the OpenAI GitHub repository.
- The data analysis pipeline relies on a vendored copy of [DOVER](https://github.com/QualityAssessment/DOVER) under `data_analysis/DOVER`, which has its own `requirements.txt` (install it only if you plan to re-run the VQA analysis).
- Pretrained backbones (TimeSformer, VideoMAEv2, DINOv2/v3, DeMamba, etc.) are downloaded automatically from Hugging Face / timm on first use. AIDE additionally requires the two backbone files already provided in `ckpts/` (`resnet50-19c8e357.pth`, `open_clip_pytorch_model.bin`).

## Dataset

### Composition

| Split | Real sources | Fake sources |
| --- | --- | --- |
| Train | Kinetics-400 (Biased) / InternVid-AES (Aligned & Expanded), Youku, OpenVidHD, Vript | Pika, OpenSora, DynamicCrafter |
| Val | Kinetics-400, InternVid-AES, Youku, OpenVidHD | SEINE (+ candidate generators for expansion: Pika, OpenSora, SD, SVD, I2VGEN_XL, DynamicCrafter, Latte, VideoCrafter) |
| Test | InternVid-AES, RealVSR, MSR-VTT, Youku, Kinetics-400, Vript, HD-VG-130M, OpenVidHD, UltraVideo | ModelScope, MorphStudio, MoonValley, Show_1, Gen2, Crafter, Lavie, Sora, WildScrape |

The exact split used in our experiments is defined by the text files in `assets/split/` (`<label>/<source>/{train,val,test}_ids.txt`). **These files are part of the benchmark** — please keep them unchanged for reproducibility.

### Download

> **TODO (authors)**: fill in download links before release.

- Real videos: collected from the official releases of Kinetics-400, InternVid, Youku, OpenVidHD, Vript, RealVSR, MSR-VTT, HD-VG-130M and UltraVideo. Links: *TBD*.
- InternVid-AES subset: to guarantee reproducibility, we will release our own crawled version (selected with `assets/RealDist-InternVid-18M-aes.jsonl`) on Hugging Face. Link: *TBD*.
- Fake videos: generated with 19 open-source / commercial generation models. Links: *TBD*.

### Directory layout

By default the configs expect the dataset at `../Data/RDL` relative to the repository root. We recommend creating a symbolic link there pointing to your actual data location (e.g. `ln -s /path/to/Data/RDL ../Data/RDL`); alternatively, override the path at runtime with `data.data_path=...`. Organize it as follows:

```
Data/RDL/
├── split/                        # copy of assets/split (train/val/test id lists)
│   ├── real/<Source>/{train,val,test}_ids.txt
│   └── fake/<Model>/{train,val,test}_ids.txt
├── video/                        # raw videos you downloaded
│   ├── real/<Source>/*.mp4
│   └── fake/<Model>/*.mp4
├── video_frames/                 # extracted by extract_frame.py
│   └── <label>/<Source>/<mode>/<video_id>/frameXXXX.jpg
├── dct_features/                 # extracted by aide_preprocess.py (AIDE only)
└── nsgvd_frames/                 # extracted on-the-fly by the NSG-VD dataloader
```

## Preprocessing

### Frame extraction (required for all models)

`extract_frame.py` reads the split files and extracts every frame of each listed video into `video_frames/` (multi-process, skips already extracted videos):

```bash
python extract_frame.py   # uses data_path='../Data/RDL' by default
```

Edit the `data_path` argument in `extract_frame.py` if your dataset lives elsewhere.

### AIDE: DCT feature extraction

AIDE additionally requires DCT features of the frames. The image-path lists are provided in `assets/aide_image_paths/` (relative to the repo root). Run:

```bash
python aide_preprocess.py            # batch extraction, 4 workers by default
# or for a single list:
python aide_extract_dct_feature.py assets/aide_image_paths/<Source>_<mode>.txt
```

Features are saved as `.pt` files under `../Data/RDL/dct_features/`.

### NSG-VD

No manual preprocessing is needed: the NSG-VD dataloader extracts the required fixed-length frame clips into `nsgvd_frames/` on first use.

## Training & Evaluation

Ready-to-use scripts for every detector and quality alignment setting live in `scripts/` (e.g. `scripts/videomaev2/biased.sh`, `scripts/aide/biased.sh`). `scripts/latest_template.sh` documents the pattern:

```bash
# Train
python train.py \
    --config-path "configs/${FEATURE_TYPE}-classifier/${MODEL}" \
    --config-name standard.yaml \
    experiment_name="${EXP_NAME}" \
    data.train_real_model="${TRAIN_REAL_MODEL}" \
    data.train_fake_model="Pika" \
    data.val_real_model="${VAL_REAL_MODEL}" \
    data.val_fake_model="SEINE" \
    data.data_expansion="${DATA_EXPANSION}" \
    log_path="./results/logs/${FEATURE_TYPE}-classifier" \
    save_ckpt_dir="./results/ckpts/${FEATURE_TYPE}-classifier/${EXP_NAME}/"

# Evaluate on all test sets
python -W ignore test.py \
    --config-path "configs/${FEATURE_TYPE}-classifier/${MODEL}" \
    --config-name test.yaml \
    experiment_name="${EXP_NAME}" \
    ckpt_path="./results/ckpts/${FEATURE_TYPE}-classifier/${EXP_NAME}/best_auroc_ckpt.pth" \
    log_path="./results/test/${FEATURE_TYPE}-classifier"
```

- `${MODEL}` ∈ `timesformer-{ssv2,k400,scratch}`, `videomaev2`, `demamba`, `dinov2`, `dinov3`, `npr`, `safe`, `aide`; `${FEATURE_TYPE}` is `video` for video-level models and `image` for frame-level ones (see `configs/`).
- Quality alignment settings only change the real training/validation sources: Biased → Kinetics-400; Aligned → InternVid-AES; Expanded → InternVid-AES + `data.data_expansion=True`.
- Training logs go to `results/logs/`, checkpoints to `results/ckpts/`, and per-source test metrics to `results/test/`.
- Each test run saves three CSVs under `results/test/.../<exp_name>/`: `<dataset>.csv` (all real/fake pairs), `<dataset>-avg.csv` (per-real-source averages), and `<dataset>-matrix.csv` — an AUROC matrix over the six real test sets Youku-mPLUG, MSR-VTT, Vript, HD-VG-130M, RealVSR and UltraVideo (rows: fake generators, columns: real sources, with row/column means).

### NSG-VD (standalone scripts)

NSG-VD uses its own entry points and configs (`configs/nsg-vd-224x224/`):

```bash
# Train
python train_nsgvd.py --config-path configs/nsg-vd-224x224 --config-name standard.yaml \
    experiment_name="standard-Pika-d"

# Test
python test_nsgvd.py --config-path configs/nsg-vd-224x224 --config-name test.yaml \
    ckpt_path="ckpts/standard-Pika-d.pth"
```

Pretrained NSG-VD checkpoints from our experiments (`standard/unbalance` × `Pika/SEINE`, `-d` / `-mp` variants) are provided in `ckpts/`. The test script saves the same three CSVs as `test.py`, including the cross-dataset AUROC matrix.

## Data Analysis

`data_analysis/` contains the analysis code used in the paper:

- **VQA scoring** with a vendored DOVER (`data_analysis/DOVER/`): download `DOVER.pth` into `data_analysis/DOVER/pretrained_weights/`, then run `vqa_predict.py` → `vqa_statistics.py` → `vqa_chart.py`; results are saved to `data_analysis/VQA_results/`.
- **Candidate expansion analysis**: `candidates_vqa_predict.py` → `candidates_vqa_statistics.py` → `candidates_vqa_chart.py` (results in `data_analysis/candidate_VQA_results/`).
- **Training data distribution**: `draw_train_data_dist.py` (figures in `data_analysis/train_dist/`).

See `data_analysis/README.md` for details.

## Project Structure

```
RDL-clean/
├── assets/
│   ├── split/                        # benchmark split files (train/val/test ids)
│   ├── aide_image_paths/             # frame path lists for AIDE DCT extraction
│   └── RealDist-InternVid-18M-aes.jsonl  # metadata of our InternVid-AES subset
├── ckpts/                            # AIDE backbones + pretrained NSG-VD checkpoints
├── configs/                          # hydra configs
│   ├── video-classifier/             # video-level detectors (VideoMAEv2, DeMamba, TimeSformer)
│   ├── image-classifier/             # frame-level detectors (DINOv2/v3, NPR, SAFE, AIDE)
│   ├── classifier-224x224/           # legacy unified configs
│   └── nsg-vd-224x224/               # NSG-VD configs
├── data/                             # datasets, splits, frame extraction
├── data_analysis/                    # VQA scoring, sampling, distribution charts (+ DOVER)
├── libs/eps_ad/                      # vendored diffusion-purification code (SAFE)
├── models/                           # detector implementations
├── results/                          # generated at runtime (auto-created, not tracked)
│   ├── ckpts/                        # checkpoints: {image,video}-classifier/<exp_name>/{best_auroc,best_f1,final}_ckpt.pth
│   ├── logs/                         # training logs: {image,video}-classifier/<exp_name>/ (loguru + tensorboard)
│   ├── test/                         # test results: {image,video}-classifier/<exp_name>/ (RDL.csv, RDL-avg.csv, RDL-matrix.csv)
│   └── outputs/                      # hydra run directories
├── scripts/                          # per-model train/eval shell scripts
├── utils/                            # training / data / MMD utilities
├── extract_frame.py                  # frame extraction (main preprocessing)
├── aide_preprocess.py                # batch DCT feature extraction for AIDE
├── aide_extract_dct_feature.py       # single-list DCT feature extraction
├── train.py / test.py                # unified training / evaluation entry points
├── train_nsgvd.py / test_nsgvd.py    # NSG-VD entry points
├── inference.py                      # single-video inference (TimeSformer)
└── requirements.txt
```

## Citation

```bibtex
@inproceedings{fangreal,
  title={Real Data Lies: Unveiling and Closing the Quality Shortcut in Generalizable AI-Generated Video Detection},
  author={Fang, Ziyuan and Wei, Tianyi and Wang, Guanjie and Zhang, Weiming and Yu, Nenghai and Zhou, Wenbo},
  booktitle={Forty-third International Conference on Machine Learning}
}
```

## License

This project is released under the [Apache License 2.0](LICENSE).
