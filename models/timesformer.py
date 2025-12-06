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
                 model_name: Literal['timesformer-k400', 'timesformer-ssv2'] = 'timesformer-k400',
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
        if model_name == 'timesformer-k400':
            self.model_name = "facebook/timesformer-base-finetuned-k400"
        elif model_name == 'timesformer-ssv2':
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
        # return self.model(pixel_values=x).logits
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
    # # 查看processor的实际类型
    # print(f"Processor的实际类型: {type(model.processor)}")
    # print(f"Processor的类: {model.processor.__class__}")

    # # 查看类继承链
    # print("\n类继承链:")
    # for i, cls in enumerate(model.processor.__class__.__mro__):
    #     print(f"{i}: {cls}")
    from transformers.models.videomae.image_processing_videomae import VideoMAEImageProcessor
    summary(model)
    model = model.cuda()
    tensor = torch.tensor(np.random.rand(1, 8, 3, 224, 224), dtype=torch.float32).cuda()
    with torch.no_grad():
        logits, attentions = model(tensor, output_attentions=True)
    
    from PIL import Image
    import random
    crop_size=(224, 224)
    frame_paths = [f'/data1/Data_AIGVDetect/GenVideo/video_frames/real/Kinetics-400/train/__mANAQ3Vts_000000_000010/frame{str(i).zfill(4)}.jpg' for i in range(1,9)]
    video_data = []
    for frame_path in frame_paths:
        img = Image.open(frame_path).convert('RGB')
        video_data.append(img)
        # video_data.append(np.array(img))
    
    width, height = video_data[0].size
    i = random.randint(0, height - crop_size[1])
    j = random.randint(0, width - crop_size[0])
    
    video_data = [np.array(img.crop((j, i, j + crop_size[0], i + crop_size[1]))) for img in video_data]
    
    processor = model.processor
    processor.do_resize = False
    processor.do_center_crop = False
    video = model.processor(images=video_data, return_tensors="pt").pixel_values[0]
    from torchvision.utils import save_image
    import os
    def save_tensor_as_images(tensor, output_dir, prefix="frame"):
        """
        将tensor保存为图片
        
        Args:
            tensor: torch.Tensor, 形状为 [8, 3, 224, 224]
            output_dir: 输出目录
            prefix: 文件名前缀
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 确保tensor在0-1范围内，如果不在则归一化
        if tensor.max() > 1.0:
            tensor = tensor.clone()  # 避免修改原始tensor
            # 反标准化：先乘以std，再加上mean
            mean = torch.tensor([0.45, 0.45, 0.45]).view(1, 3, 1, 1)
            std = torch.tensor([0.225, 0.225, 0.225]).view(1, 3, 1, 1)
            tensor = tensor * std + mean
            # 限制到0-1范围
            tensor = torch.clamp(tensor, 0, 1)
        
        # 保存每一帧
        for i in range(tensor.size(0)):
            filename = os.path.join(output_dir, f"random_crop_{prefix}_{i:02d}.png")
            save_image(tensor[i], filename)
            print(f"保存: {filename}")
        
        print(f"共保存了 {tensor.size(0)} 张图片到 {output_dir}")
    save_tensor_as_images(video, "./results/timesformer_process", "video")
    breakpoint()