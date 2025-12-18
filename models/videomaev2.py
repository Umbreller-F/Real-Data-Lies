from transformers import VideoMAEImageProcessor, AutoModel, AutoConfig
from loguru import logger
from typing import Literal
from copy import deepcopy
import numpy as np
import torch.nn as nn
import torch
import logging

logging.getLogger("transformers_modules.OpenGVLab.VideoMAEv2-Base").setLevel(logging.WARNING)


class VideoMAEv2Classifier(nn.Module):
    def __init__(self,
                 model_name: Literal['VideoMAEv2-Base', 'VideoMAEv2-Large'] = 'VideoMAEv2-Base',
                 output_dim: int = 1,
                 dropout_rate: float = 0.1):
        super().__init__()
        config = AutoConfig.from_pretrained(f"OpenGVLab/{model_name}", trust_remote_code=True, local_files_only=True)
        self._processor = VideoMAEImageProcessor.from_pretrained(f"OpenGVLab/{model_name}", local_files_only=True)
        self.model = AutoModel.from_pretrained(f"OpenGVLab/{model_name}", config=config, trust_remote_code=True, local_files_only=True)
        embed_dim = self.model.config.model_config['embed_dim']
        # Replace classification head with custom binary classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x, output_attentions: bool = False):
        embeddings = self.model(pixel_values=x.permute(0, 2, 1, 3, 4))
        logits = self.classifier(embeddings)
        return logits
    
    @property
    def processor(self):
        return deepcopy(self._processor)


if __name__ == "__main__":
    from torchinfo import summary
    logger.debug("This module is not meant to be run directly. Import it in your code to use the models.")
    # model_type = "Base"
    # config = AutoConfig.from_pretrained(f"OpenGVLab/VideoMAEv2-{model_type}", trust_remote_code=True, local_files_only=True)
    # processor = VideoMAEImageProcessor.from_pretrained(f"OpenGVLab/VideoMAEv2-{model_type}", local_files_only=True)
    # model = AutoModel.from_pretrained(f"OpenGVLab/VideoMAEv2-{model_type}", config=config, trust_remote_code=True, local_files_only=True)
    # summary(model)
    # print(processor)

    # # video = list(np.random.rand(16, 3, 224, 224))
    # video = [np.random.randint(0, 256, (3, 224, 224), dtype=np.uint8) for _ in range(16)]

    # # B, T, C, H, W -> B, C, T, H, W
    # inputs = processor(video, return_tensors="pt")
    # breakpoint()
    # inputs['pixel_values'] = inputs['pixel_values'].permute(0, 2, 1, 3, 4)

    # with torch.no_grad():
    #     outputs = model(**inputs)
    # print("Logits shape:", outputs.shape)
    model = VideoMAEv2Classifier()
    summary(model)
    model = model.cuda()
    tensor = torch.tensor(np.random.rand(4, 16, 3, 224, 224), dtype=torch.float32).cuda()
    with torch.no_grad():
        logits = model(tensor)
    print("Logits shape:", logits.shape)
