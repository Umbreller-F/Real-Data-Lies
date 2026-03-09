import logging
import torch
import torch.nn as nn
import numpy as np
from typing import Literal
from torchinfo import summary
from loguru import logger
from copy import deepcopy
from transformers import AutoImageProcessor, AutoModel

logging.getLogger("dinov2").setLevel(logging.WARNING)
logging.getLogger("dinov3").setLevel(logging.WARNING)


class DINOv2(nn.Module):
    def __init__(self,
                 model_name: Literal['dinov2-base', 'dinov2-large'] = 'dinov2-large',
                 output_dim: int = 1,
                 dropout_rate: float = 0.1):
        super().__init__()
        self._processor = AutoImageProcessor.from_pretrained(f'facebook/{model_name}', local_files_only=True)
        self.model = AutoModel.from_pretrained(f'facebook/{model_name}', local_files_only=True)
        embed_dim = self.model.config.hidden_size
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
    
    def forward(self, x):
        embeddings = self.model(pixel_values=x).pooler_output
        logits = self.classifier(embeddings)
        return logits
    
    @property
    def processor(self):
        return deepcopy(self._processor)


class DINOv3(nn.Module):
    def __init__(self,
                 model_name: Literal['dinov3-vitb16', 'dinov3-vitl16', 'dinov3-convnext-base', 'dinov3-convnext-large'] = 'dinov3-vitl16',
                 output_dim: int = 1,
                 dropout_rate: float = 0.1):
        super().__init__()
        self._processor = AutoImageProcessor.from_pretrained(f'facebook/{model_name}-pretrain-lvd1689m', local_files_only=True)
        self.model = AutoModel.from_pretrained(f'facebook/{model_name}-pretrain-lvd1689m', local_files_only=True)
        if model_name in ['dinov3-convnext-base', 'dinov3-convnext-large']:
            if model_name == 'dinov3-convnext-base':
                embed_dim = 1024
            else:
                embed_dim = 1536
        else:
            embed_dim = self.model.config.hidden_size
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
    
    def forward(self, x):
        embeddings = self.model(pixel_values=x).pooler_output
        logits = self.classifier(embeddings)
        return logits
    
    @property
    def processor(self):
        return deepcopy(self._processor)


if __name__ == "__main__":
    logger.debug("This module is not meant to be run directly. Import it in your code to use the models.")

    # from PIL import Image
    # import requests

    # url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    # image = Image.open(requests.get(url, stream=True).raw)

    # processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base', 
    #                                                local_files_only=True
    #                                                )
    # model = AutoModel.from_pretrained('facebook/dinov2-base', 
    #                                   local_files_only=True
    #                                   )
    # summary(model)
    # print(processor)
    # raw_inputs = processor(images=image)
    # breakpoint()

    # inputs = processor(images=image, return_tensors="pt")
    # outputs = model(**inputs)
    # last_hidden_states = outputs.last_hidden_state

    # import torch
    # from transformers import AutoImageProcessor, AutoModel
    from transformers.image_utils import load_image

    # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # image = load_image(url)

    # pretrained_model_name = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    # processor = AutoImageProcessor.from_pretrained(pretrained_model_name, local_files_only=True)
    # model = AutoModel.from_pretrained(pretrained_model_name, local_files_only=True)

    # inputs = processor(images=image, return_tensors="pt").to(model.device)
    # with torch.inference_mode():
    #     outputs = model(**inputs)

    # pooled_output = outputs.pooler_output
    # print("Pooled output shape:", pooled_output.shape)

    # breakpoint()

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = load_image(url)
    pretrained_model_name = "facebook/dinov3-convnext-large-pretrain-lvd1689m"
    processor = AutoImageProcessor.from_pretrained(pretrained_model_name, token="REDACTED")
    model = AutoModel.from_pretrained(
        pretrained_model_name, 
        device_map="auto", 
        token="REDACTED"
    )
    summary(model)
    inputs = processor(images=image, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model(**inputs)

    pooled_output = outputs.pooler_output
    print("Pooled output shape:", pooled_output.shape)
    breakpoint()

    # model = DINOv3('dinov3-convnext-base')
    # # model = DINOv2()
    # summary(model)
    # model = model.cuda()
    # tensor = torch.tensor(np.random.rand(4, 3, 224, 224), dtype=torch.float32).cuda()
    # print(model(tensor).shape)
    # breakpoint()
