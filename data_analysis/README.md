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
# draw train data distribution
python ./data_analysis/draw_train_data_dist.py
# data expansion candidates
python ./data_analysis/vqa_predict.py
python ./data_analysis/vqa_statistics.py
python ./data_analysis/vqa_chart.py
```

## Output

Results will be saved in `data_analysis/VQA_results/`, `data_analysis/train_dist` and `data_analysis/candidate_VQA_results`.