from transformers import VideoMAEImageProcessor, AutoModel, AutoConfig
from loguru import logger
import numpy as np
import torch


if __name__ == "__main__":
    from torchinfo import summary
    logger.debug("This module is not meant to be run directly. Import it in your code to use the models.")
    config = AutoConfig.from_pretrained("OpenGVLab/VideoMAEv2-Base", trust_remote_code=True, local_files_only=True)
    processor = VideoMAEImageProcessor.from_pretrained("OpenGVLab/VideoMAEv2-Base", local_files_only=True)
    model = AutoModel.from_pretrained('OpenGVLab/VideoMAEv2-Base', config=config, trust_remote_code=True, local_files_only=True)
    summary(model)
    print(processor)

    # video = list(np.random.rand(16, 3, 224, 224))
    video = [np.random.randint(0, 256, (3, 224, 224), dtype=np.uint8) for _ in range(16)]

    # B, T, C, H, W -> B, C, T, H, W
    inputs = processor(video, return_tensors="pt")
    inputs['pixel_values'] = inputs['pixel_values'].permute(0, 2, 1, 3, 4)

    with torch.no_grad():
        outputs = model(**inputs)
    print("Logits shape:", outputs.shape)
