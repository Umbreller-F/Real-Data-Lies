from transformers import AutoImageProcessor, TimesformerForVideoClassification
from torchinfo import summary
from typing import Literal
from loguru import logger
import numpy as np
import torch.nn as nn
import torch


class TimesformerBinaryClassifier(nn.Module):
    def __init__(self, 
                #  model_name: str = "facebook/timesformer-base-finetuned-ssv2",
                model_name: Literal['timesformer-k400', 'timesformer-ssv2'],
                 num_classes: int = 2,
                 freeze_backbone: bool = False,
                 dropout_rate: float = 0.1):
        """
        Timesformer video binary classification model
        
        Args:
            model_name: Pretrained model name
            num_classes: Number of classes (set to 2 for binary classification)
            freeze_backbone: Whether to freeze backbone and only train classifier head
            dropout_rate: Dropout rate for classifier
        """
        super().__init__()
        
        # Load pretrained model and processor
        if model_name == 'timesformer-k400':
            self.model_name = "facebook/timesformer-base-finetuned-k400"
        elif model_name == 'timesformer-ssv2':
            self.model_name = "facebook/timesformer-base-finetuned-ssv2"
        else:
            raise ValueError(f"Unknown model name: {model_name}")
        self.model = TimesformerForVideoClassification.from_pretrained(self.model_name)
        self.processor = AutoImageProcessor.from_pretrained(self.model_name, use_fast=False)

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
            nn.Linear(128, num_classes)
        )
        
        # Freeze backbone if specified
        if freeze_backbone:
            self.freeze_backbone()
        
        self.num_classes = num_classes
        
    def freeze_backbone(self):
        """Freeze Timesformer backbone, only train classifier head"""
        for param in self.model.timesformer.parameters():
            param.requires_grad = False
        logger.info("Backbone frozen, only classifier head will be trained")
    
    def forward(self, x):
        # inputs = self.processor(images=x, return_tensors="pt")
        return self.model(pixel_values=x).logits


if __name__ == "__main__":
    print("This module is not meant to be run directly. Import it in your code to use the models.")
    video = list(np.random.randint(0, 256, (8, 224, 224, 3), dtype=np.uint8))

    processor = AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-ssv2", use_fast=False)
    model = TimesformerForVideoClassification.from_pretrained("facebook/timesformer-base-finetuned-ssv2")

    inputs = processor(images=video, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    predicted_class_idx = logits.argmax(-1).item()
    print("Predicted class:", model.config.id2label[predicted_class_idx])
    summary(model)

    tensor = torch.tensor(np.random.rand(32, 8, 3, 224, 224), dtype=torch.float32).cuda()
    model = TimesformerBinaryClassifier(model_name='timesformer-k400', freeze_backbone=False)
    model = model.cuda()
    with torch.no_grad():
        print(model(tensor).shape)
    print(model.processor)
