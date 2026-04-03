# Video Quality Assessment with DOVER

## Setup

1. Download the checkpoint file from [DOVER](https://github.com/QualityAssessment/DOVER/releases/download/v0.1.0/DOVER.pth).
2. Place it in `data_analysis/DOVER/pretrained_weights/`.

## Run

Execute the following commands in order:

```bash
python ./data_analysis/vqa_predict.py
python ./data_analysis/vqa_statistics.py
python ./data_analysis/vqa_chart.py
```

## Output

Results will be saved in `data_analysis/VQA_results/`.