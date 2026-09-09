<h1 align="center">
     Work in Progress
</h1>

```bash
nohup bash scripts/dinov3/biased-dinov3.sh > nohup_logs/$(date +%Y%m%d_%H%M%S)_biased_dinov3.log 2>&1 &
```

AIDE pretrained weights：

open_clip_pytorch_model.bin : https://huggingface.co/laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-soup/tree/main

resnet50: https://download.pytorch.org/models/resnet50-19c8e357.pth (already in repo)