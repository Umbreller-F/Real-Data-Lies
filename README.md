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

# Install OpenAI CLIP from GitHub (must be built without build isolation:
# its setup.py imports pkg_resources, which setuptools>=81 — used by pip's
# isolated build env — no longer provides)
pip install "clip @ git+https://github.com/openai/CLIP.git@dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1" --no-build-isolation
```

Notes:

- `requirements.txt` was exported from the authors' environment and pruned to the packages actually imported by this project. `clip` is installed separately from the OpenAI GitHub repository as shown above.
- The data analysis pipeline relies on a vendored copy of [DOVER](https://github.com/QualityAssessment/DOVER) under `data_analysis/DOVER`, which has its own `requirements.txt` (install it only if you plan to re-run the VQA analysis).
- If `huggingface.co` is unreachable from your network, set `HF_ENDPOINT=https://hf-mirror.com` before running the scripts so automatic downloads go through the mirror.

## Pretrained Weights

Most detectors download their backbones automatically from Hugging Face or PyTorch Hub on first use (cached under `~/.cache/huggingface` / `~/.cache/torch`):

| Detector | Pretrained weights | Source | Download |
| --- | --- | --- | --- |
| TimeSformer-ssv2 | `facebook/timesformer-base-finetuned-ssv2` | Hugging Face | automatic |
| TimeSformer-k400 | `facebook/timesformer-base-finetuned-k400` | Hugging Face | automatic |
| TimeSformer-scratch | — (random initialization) | — | — |
| VideoMAEv2 | `OpenGVLab/VideoMAEv2-Base` | Hugging Face | automatic |
| DeMamba | `microsoft/xclip-base-patch16` | Hugging Face | automatic |
| DINOv2 | `facebook/dinov2-large` | Hugging Face | automatic |
| DINOv3 | `facebook/dinov3-vitl16-pretrain-lvd1689m` | Hugging Face (**gated**) | automatic, after accepting the license on the model page and logging in with `huggingface-cli login` |
| NPR | — (custom truncated ResNet-style backbone, trained from scratch) | — | — |
| SAFE | — (custom truncated ResNet-style backbone, trained from scratch) | — | — |
| AIDE | ResNet-50 + OpenCLIP ConvNeXt-XXLarge | see below | **manual** |
| NSG-VD | guided-diffusion `256x256_diffusion_uncond.pt` (feature extractor only; discriminator trained from scratch) | [OpenAI guided-diffusion](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt) | **manual** → `../Checkpoints/`; our trained NSG-VD checkpoints are provided in `ckpts/` for evaluation |

AIDE is the only detector that requires manually downloaded backbones. These are the two files officially specified by the AIDE author (see [AIDE issue #8](https://github.com/shilinyan99/AIDE/issues/8)); we do **not** bundle them in this repo — please download them yourself and place both under `ckpts/`:

| File | Official source |
| --- | --- |
| `ckpts/resnet50-19c8e357.pth` | https://download.pytorch.org/models/resnet50-19c8e357.pth |
| `ckpts/open_clip_pytorch_model.bin` | https://huggingface.co/laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-soup/tree/main |

```bash
wget -P ckpts https://download.pytorch.org/models/resnet50-19c8e357.pth
wget -P ckpts https://huggingface.co/laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-soup/resolve/main/open_clip_pytorch_model.bin
# or via the mirror: https://hf-mirror.com/laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-soup/resolve/main/open_clip_pytorch_model.bin
```

## Dataset

### Composition

| Split | Real sources | Fake sources |
| --- | --- | --- |
| Train | Kinetics-400 (Biased) / InternVid-AES (Aligned & Expanded), Youku, OpenVidHD, Vript | Pika, OpenSora, DynamicCrafter |
| Val | Kinetics-400, InternVid-AES, Youku, OpenVidHD | SEINE (+ candidate generators for expansion: Pika, OpenSora, SD, SVD, I2VGEN_XL, DynamicCrafter, Latte, VideoCrafter) |
| Test | InternVid-AES, RealVSR, MSR-VTT, Youku, Kinetics-400, Vript, HD-VG-130M, OpenVidHD, UltraVideo | ModelScope, MorphStudio, MoonValley, Show_1, Gen2, Crafter, Lavie, Sora, WildScrape |

The exact split used in our experiments is defined by the text files in `assets/split/` (`<label>/<source>/{train,val,test}_ids.txt`). **These files are part of the benchmark** — please keep them unchanged for reproducibility. For each real/fake source we use at most **10,000 videos for training** and **1,000 for validation / testing**; sources with fewer available videos are used in full (e.g. the Sora test set has 56 videos).

### Download

| Data | Link | Provider |
| --- | --- | --- |
| **Real** | | |
| Kinetics-400 | [kinetics-dataset](https://github.com/cvdfoundation/kinetics-dataset) | Official release |
| InternVid-AES | [umbreller/RDL-InternVid-AES](https://huggingface.co/datasets/umbreller/RDL-InternVid-AES) | **Released by us** (academic research only) |
| Youku-mPLUG | [GenVideo](https://modelscope.cn/datasets/cccnju/Gen-Video) | Re-distributed via the GenVideo benchmark |
| OpenVidHD | [OpenVid-1M](https://huggingface.co/datasets/nkp37/OpenVid-1M) | Official release |
| Vript | [GenVidBench](https://huggingface.co/datasets/jian-0/GenVidBench) | Re-distributed via the GenVidBench benchmark |
| RealVSR | [RealVSR](https://github.com/IanYeung/RealVSR) | Official release |
| MSR-VTT | [GenVideo](https://modelscope.cn/datasets/cccnju/Gen-Video) | Re-distributed via the GenVideo benchmark |
| HD-VG-130M | [GenVidBench](https://huggingface.co/datasets/jian-0/GenVidBench) | Re-distributed via the GenVidBench benchmark |
| UltraVideo | [UltraVideo](https://huggingface.co/datasets/APRIL-AIGC/UltraVideo) | Official release |
| **Fake** | | |
| 19 generators (Pika, SEINE, Sora, ...) | [GenVideo](https://modelscope.cn/datasets/cccnju/Gen-Video) | Collected and released by the GenVideo benchmark |

Note that several real sources are re-distributions via third-party benchmarks rather than the original official releases, and the fake videos were collected and released by the GenVideo benchmark (see the Provider column). The InternVid-AES subset is our own collected version (selected with `assets/RealDist-InternVid-18M-aes.jsonl`): 12,000 videos split into 20 GB tar volumes, for academic research use only — see the dataset page for extraction instructions.

### Directory layout

By default the configs expect the dataset at `../Data/RDL` relative to the repository root. We recommend creating a symbolic link there pointing to your actual data location (e.g. `ln -s /path/to/Data/RDL ../Data/RDL`); alternatively, override the path at runtime with `data.data_path=...`.

After downloading the raw videos, **copy the benchmark split files into the dataset root** so the dataloaders can find them:

```bash
cp -r assets/split ../Data/RDL/split
```

Then copy the videos listed in the split files into `video/<label>/<Source>/` — note that this directory is **flat**: train/val/test videos are stored together, with no per-split subdirectories. `extract_frame.py` reads the split files and builds the train/val/test frame directories (`video_frames/.../<mode>/`) by itself. Only the videos referenced by the split files are ever touched; any extra videos placed in the folder are simply ignored (but copying just the listed ones saves disk space).

Then organize the dataset as follows (only `split/` and `video/` are required manually; the rest are produced by the preprocessing steps below):

```
Data/RDL/
├── split/                        # copied from assets/split (train/val/test id lists)
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

AIDE additionally requires DCT features of the frames, so **run `extract_frame.py` first** — the image-path lists in `assets/aide_image_paths/` enumerate the extracted frames (the 8 sampled frames of every split-listed video) under `video_frames/`. Then run:

```bash
python aide_preprocess.py            # batch extraction over all lists, 4 workers by default
# or for a single list:
python aide_extract_dct_feature.py assets/aide_image_paths/<Source>_<mode>.txt
```

Features are saved as `.pt` files under `../Data/RDL/dct_features/`, mirroring the `video_frames/` layout. Lists whose features already exist are skipped automatically, so re-running is safe.

### NSG-VD

No manual preprocessing is needed, but the **first** NSG-VD run performs two on-the-fly extraction steps (both cached and reused afterwards):

1. fixed-length 8-frame clips into `../Data/RDL/nsgvd_frames/`;
2. diffusion-based score/velocity features into `../Data/RDL/nsg-vd/STEPS_5/` — this step needs the guided-diffusion checkpoint `256x256_diffusion_uncond.pt` at `../Checkpoints/` (see Pretrained Weights).

Expect the first full run (train + all test sets) to take a few hours on a single GPU; later runs reuse the caches and are much faster.

## Training & Evaluation

### Quick start

> **One command per experiment** — ready-to-use scripts for every detector and quality alignment setting live in `scripts/`. Each script runs **training and full test-set evaluation** end-to-end:

```bash
bash scripts/demamba/aligned.sh        # DeMamba, Aligned setting
bash scripts/timesformer-k400/biased.sh  # TimeSformer-K400, Biased setting
bash scripts/aide/expanded.sh          # AIDE, Expanded setting
```

The pattern is `scripts/<detector>/<biased|aligned|expanded>.sh`. The only exceptions are **NPR** and **NSG-VD**, which we only evaluate under the Biased setting in our paper, so only `scripts/npr/biased.sh` and `scripts/nsgvd/biased.sh` are provided (NSG-VD also uses its own entry points, see below). All scripts are plain bash — set `CUDA_VISIBLE_DEVICES` in front as needed (e.g. `CUDA_VISIBLE_DEVICES=0 bash scripts/demamba/aligned.sh`).

### Details

All scripts follow the same pattern:

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
- Quality alignment settings only change the real training/validation sources: 
  - Biased → Kinetics-400
  - Aligned → InternVid-AES
  - Expanded → InternVid-AES + `data.data_expansion=True`.

For example, DeMamba under the Aligned setting (equivalent to `bash scripts/demamba/aligned.sh`):

```bash
python train.py \
    --config-path "configs/video-classifier/demamba" \
    --config-name standard.yaml \
    experiment_name="aligned-Pika-demamba" \
    data.train_real_model="InternVid-AES" \
    data.train_fake_model="Pika" \
    data.val_real_model="InternVid-AES" \
    data.val_fake_model="SEINE" \
    data.data_expansion=False \
    log_path="./results/logs/video-classifier" \
    save_ckpt_dir="./results/ckpts/video-classifier/aligned-Pika-demamba/"

python -W ignore test.py \
    --config-path "configs/video-classifier/demamba" \
    --config-name test.yaml \
    experiment_name="aligned-Pika-demamba" \
    ckpt_path="./results/ckpts/video-classifier/aligned-Pika-demamba/best_auroc_ckpt.pth" \
    log_path="./results/test/video-classifier"
```
- Training logs go to `results/logs/`, checkpoints to `results/ckpts/`, and per-source test metrics to `results/test/`.
- Each test run saves three CSVs under `results/test/.../<exp_name>/`: `<dataset>.csv` (all real/fake pairs), `<dataset>-avg.csv` (per-real-source averages), and `<dataset>-matrix.csv` — an AUROC matrix over the six real test sets Youku-mPLUG, MSR-VTT, Vript, HD-VG-130M, RealVSR and UltraVideo (rows: fake generators, columns: real sources, with row/column means).

### NSG-VD (standalone entry points)

NSG-VD is only evaluated under the **Biased** setting in our paper and uses its own entry points and configs (`configs/nsg-vd-224x224/`). To reproduce, run:

```bash
bash scripts/nsgvd/biased.sh   # edit VARIANT="d" / "mp" at the top to choose the MMD variant
```

which is equivalent to:

```bash
# Train (variant d; use model.is_yy_zero=True for mp)
python train_nsgvd.py --config-path configs/nsg-vd-224x224 --config-name standard.yaml \
    experiment_name="standard-Pika-d" \
    model.is_yy_zero=False

# Test
python test_nsgvd.py --config-path configs/nsg-vd-224x224 --config-name test.yaml \
    experiment_name="standard-Pika-d" \
    model.is_yy_zero=False \
    ckpt_path="./results/ckpts/nsg-vd/standard-Pika-d/best_ckpt.pth" \
    log_path="./results/test/nsg-vd/standard-Pika-d"
```

NSG-VD requires no external pretrained backbone: the discriminator is trained from scratch. Its score/velocity feature extractor does need the guided-diffusion ImageNet 256×256 unconditional model — download [256x256_diffusion_uncond.pt](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt) and place it at `../Checkpoints/256x256_diffusion_uncond.pt` (relative to the repository root) before the first run. Pretrained NSG-VD checkpoints from our experiments (`standard/unbalance` × `Pika/SEINE`, `-d` / `-mp` variants) are provided in `ckpts/` for direct evaluation. The test script saves the same three CSVs as `test.py`, including the cross-dataset AUROC matrix.

Notes:

- **Variants**: `-d` (`model.is_yy_zero=False`) and `-mp` (`model.is_yy_zero=True`) are the two MMD variants reported in the paper; choose one via `VARIANT` in `scripts/nsgvd/biased.sh` (default `d`).
- **First run is slow**: the dataloader first extracts 8-frame clips into `Data/RDL/nsgvd_frames/`, then computes diffusion-based score/velocity features for every video, cached under `Data/RDL/nsg-vd/STEPS_5/<label>/<source>/<split>/`. Both caches are reused across variants, reruns and evaluation, so subsequent runs are much faster.
- **Evaluating the provided checkpoints** directly (no training):

```bash
python test_nsgvd.py --config-path configs/nsg-vd-224x224 --config-name test.yaml \
    experiment_name="standard-Pika-d" \
    model.is_yy_zero=False \
    ckpt_path="ckpts/standard-Pika-d.pth" \
    log_path="./results/test/nsg-vd/standard-Pika-d-provided"
```

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
├── scripts/                          # per-detector train/eval shell scripts (biased / aligned / expanded)
├── utils/                            # training / data / MMD utilities
├── extract_frame.py                  # frame extraction (main preprocessing)
├── aide_preprocess.py                # batch DCT feature extraction for AIDE
├── aide_extract_dct_feature.py       # single-list DCT feature extraction
├── train.py / test.py                # unified training / evaluation entry points
├── train_nsgvd.py / test_nsgvd.py    # NSG-VD entry points
├── inference.py                      # single-video inference (TimeSformer)
└── requirements.txt
```

## Acknowledgements

Our benchmark is built upon data from the following sources — we sincerely thank all of them:

- [GenVideo](https://github.com/chenhaoxing/DeMamba) (released together with the DeMamba detector)
- [GenVidBench](https://github.com/genvidbench/GenVidBench)
- [MSR-VTT](https://www.microsoft.com/en-us/research/publication/msr-vtt-a-large-video-description-dataset-for-bridging-video-and-language/)
- [Kinetics-400](https://github.com/cvdfoundation/kinetics-dataset)
- [InternVid](https://github.com/OpenGVLab/InternVideo/tree/main/Data/InternVid)
- [Youku-mPLUG](https://github.com/X-PLUG/Youku-mPLUG)
- [Vript](https://github.com/mutonix/Vript)
- [OpenVid-1M](https://github.com/NJU-PCALab/OpenVid-1M)
- [RealVSR](https://github.com/IanYeung/RealVSR)
- [UltraVideo](https://github.com/xzc-zju/UltraVideo)
- [HD-VG-130M](https://github.com/daooshee/HD-VG-130M)

We also thank the following open-source projects, from which our detector implementations and analysis tools are adapted:

- [DeMamba](https://github.com/chenhaoxing/DeMamba) (released together with the GenVideo benchmark)
- [NSG-VD](https://github.com/ZSHsh98/NSG-VD)
- [DOVER](https://github.com/QualityAssessment/DOVER)
- [NPR](https://github.com/chuangchuangtan/NPR-DeepfakeDetection)
- [SAFE](https://github.com/Ouxiang-Li/SAFE)
- [AIDE](https://github.com/shilinyan99/AIDE)
- [DINOv2](https://github.com/facebookresearch/dinov2)
- [DINOv3](https://github.com/facebookresearch/dinov3)
- [VideoMAEv2](https://github.com/OpenGVLab/VideoMAEv2)
- [TimeSformer](https://github.com/facebookresearch/TimeSformer)

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
