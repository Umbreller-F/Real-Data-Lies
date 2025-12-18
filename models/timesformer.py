from transformers import AutoImageProcessor, TimesformerForVideoClassification, TimesformerConfig
from torchinfo import summary
from typing import Literal
from copy import deepcopy
from loguru import logger
import numpy as np
import torch.nn as nn
import torch


class TimesformerBinaryClassifier(nn.Module):
    def __init__(self,
                 model_name: Literal['TimeSformer-k400', 'TimeSformer-ssv2'] = 'TimeSformer-k400',
                 pretrained: bool = True,
                 output_dim: int = 1,
                 freeze_backbone: bool = False,
                 dropout_rate: float = 0.1):
        """
        Timesformer video binary classification model
        
        Args:
            model_name: Pretrained model name
            pretrained: Whether to use pretrained weights
            output_dim: Dimension of output (1 for binary classification)
            freeze_backbone: Whether to freeze backbone and only train classifier head
            dropout_rate: Dropout rate for classifier
        """
        super().__init__()
        
        # Load model and processor
        if model_name == 'TimeSformer-k400':
            self.model_name = "facebook/timesformer-base-finetuned-k400"
        elif model_name == 'TimeSformer-ssv2':
            self.model_name = "facebook/timesformer-base-finetuned-ssv2"
        else:
            raise ValueError(f"Unknown model name: {model_name}")
        
        if pretrained:
            logger.info(f"Loading pretrained Timesformer model: {self.model_name}.")
            self.model = TimesformerForVideoClassification.from_pretrained(self.model_name, local_files_only=True)
        else:
            logger.info(f"Initializing Timesformer model from scratch.")
            config = TimesformerConfig.from_pretrained(self.model_name, local_files_only=True)
            self.model = TimesformerForVideoClassification(config)
        
        self._processor = AutoImageProcessor.from_pretrained(self.model_name, use_fast=False, local_files_only=True)

        # Get original feature dimension
        original_hidden_size = self.model.config.hidden_size
        
        # Replace classification head with custom binary classifier
        self.model.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(original_hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        
        # Freeze backbone if specified
        if freeze_backbone:
            self.freeze_backbone()
        
    def freeze_backbone(self):
        """Freeze Timesformer backbone, only train classifier head"""
        for param in self.model.timesformer.parameters():
            param.requires_grad = False
        logger.info("Backbone frozen, only classifier head will be trained")
    
    def forward(self, x, output_attentions: bool = False):
        outputs = self.model(pixel_values=x, output_attentions=output_attentions)
        if output_attentions:
            return outputs.logits, outputs.attentions
        else:
            return outputs.logits
    
    @property
    def processor(self):
        return deepcopy(self._processor)


if __name__ == "__main__":
    logger.debug("This module is not meant to be run directly. Import it in your code to use the models.")
    model = TimesformerBinaryClassifier(model_name='timesformer-k400', freeze_backbone=False)
    from transformers.models.videomae.image_processing_videomae import VideoMAEImageProcessor
    summary(model)
    model = model.cuda()
    tensor = torch.tensor(np.random.rand(1, 8, 3, 224, 224), dtype=torch.float32).cuda()
    with torch.no_grad():
        logits, attentions = model(tensor, output_attentions=True)
    print("Logits shape:", logits.shape)
