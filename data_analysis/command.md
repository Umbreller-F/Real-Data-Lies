```bash
# Stratified sampling by quality scores
python ./data_analysis/lsvq_sample_train_val.py
python ./data_analysis/lsvq_sample_test_1080p.py
# Optional: Generate statistical charts for each part of LSVQ
python ./data_analysis/lsvq_statistics.py
# Copy sampled videos to RealDist
python ./data_analysis/lsvq_to_realdist.py
# Sample Youku videos for RealDist
python ./data_analysis/youku_sample.py
# VQA analysis
python ./data_analysis/vqa_sample.py
python ./data_analysis/vqa_predict.py
python ./data_analysis/vqa_statistics.py
python ./data_analysis/vqa_chart.py
```