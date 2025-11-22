import torch
import torch.nn as nn
import numpy as np
from typing import Literal
from torchinfo import summary
from loguru import logger


class DINOv2WithLinearProbe(nn.Module):
    def __init__(
        self,
        model_name: Literal[
            'dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14',
            'dinov2_vits14_reg', 'dinov2_vitb14_reg', 'dinov2_vitl14_reg', 'dinov2_vitg14_reg'
        ],
        freeze_backbone: bool = True,
        num_layers_to_use: int = None
    ):
        """DINOv2 model with a binary classification head."""
        super().__init__()
        
        # Load pretrained DINOv2 model
        self.backbone = torch.hub.load(
            '/home/ziyuanfang/.cache/torch/hub/facebookresearch_dinov2_main',
            model=model_name,
            source='local'
        )
        
        # Determine embedding dimension based on model type
        if 'vits14' in model_name:
            embed_dim = 384
            num_blocks = 12  # Small model has 12 transformer blocks
        elif 'vitb14' in model_name:
            embed_dim = 768
            num_blocks = 12  # Base model has 12 transformer blocks
        elif 'vitl14' in model_name:
            embed_dim = 1024
            num_blocks = 24  # Large model has 24 transformer blocks
        elif 'vitg14' in model_name:
            embed_dim = 1536
            num_blocks = 40  # Giant model has 40 transformer blocks
        else:
            raise ValueError(f"Unknown model type: {model_name}")
        
        # Store number of layers to use
        self.num_layers_to_use = num_layers_to_use if num_layers_to_use is not None else num_blocks

        # Modify the backbone to use only first n blocks
        if self.num_layers_to_use < num_blocks:
            # The transformer blocks are in backbone.blocks
            self.backbone.blocks = self.backbone.blocks[:self.num_layers_to_use]
            # Replace the norm layer with new LayerNorm
            self.backbone.norm = nn.LayerNorm(embed_dim)

        # Add binary classification head (Linear layer)
        self.backbone.head = nn.Sequential(
            nn.Linear(embed_dim, 1),
        )
        
        # Freeze backbone parameters (only train classifier head)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        predictions = self.backbone(x)
        return predictions  # [batch_size, 1]


class DINOv3WithLinearProbe(nn.Module):
    def __init__(
        self,
        model_name: Literal[
            'dinov2_vits14', 'dinov3_vits16plus', 'dinov2_vitb14', 'dinov2_vitl14',
            'dinov3_convnext_tiny', 'dinov3_convnext_small', 'dinov3_convnext_base', 'dinov3_convnext_large'
        ],
        freeze_backbone: bool = True,
        num_layers_to_use: int = None
    ):
        """DINOv3 model with a binary classification head."""
        super().__init__()
        
        # Load pretrained DINOv2 model
        self.backbone = torch.hub.load(
            '/home/ziyuanfang/.cache/torch/hub/facebookresearch_dinov3_main',
            model=model_name,
            source='local'
        )
        
        # Determine embedding dimension based on model type
        if 'vits16' in model_name:
            embed_dim = 384
            num_blocks = 12  # Small model has 12 transformer blocks
        elif 'vitb16' in model_name:
            embed_dim = 768
            num_blocks = 12  # Base model has 12 transformer blocks
        elif 'vitl16' in model_name:
            embed_dim = 1024
            num_blocks = 24  # Large model has 24 transformer blocks
        elif 'convnext_tiny' in model_name:
            embed_dim = 384
            num_blocks = None
        elif 'convnext_small' in model_name:
            embed_dim = 768
            num_blocks = None
        elif 'convnext_base' in model_name:
            embed_dim = 1024
            num_blocks = None
        elif 'convnext_large' in model_name:
            embed_dim = 1536
            num_blocks = None
        else:
            raise ValueError(f"Unknown model type: {model_name}")
        
        # Store number of layers to use
        self.num_layers_to_use = num_layers_to_use if num_layers_to_use is not None else num_blocks

        # Modify the backbone to use only first n blocks
        if self.num_layers_to_use < num_blocks and 'vit' in model_name:
            # The transformer blocks are in backbone.blocks
            self.backbone.blocks = self.backbone.blocks[:self.num_layers_to_use]
            # Replace the norm layer with new LayerNorm
            self.backbone.norm = nn.LayerNorm(embed_dim)

        # Add binary classification head (Linear layer)
        self.backbone.head = nn.Sequential(
            nn.Linear(embed_dim, 1),
        )
        
        # Freeze backbone parameters (only train classifier head)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        predictions = self.backbone(x)
        return predictions  # [batch_size, 1]


if __name__ == "__main__":
    logger.debug("This module is not meant to be run directly. Import it in your code to use the models.")

    logger.debug("Testing DINOv2WithLinearProbe model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dinov2 = DINOv2WithLinearProbe('dinov2_vitb14', freeze_backbone=False, num_layers_to_use=None)
    dinov2.to(device)
    summary(dinov2)

    input_tensor = torch.tensor(np.random.rand(32, 3, 224, 224), dtype=torch.float32).cuda()  # Example input tensor
    output = dinov2(input_tensor)  # Forward pass
    print("Output shape:", output.shape)  # Should print the shape of the output tensor

    logger.debug("Testing DINOv3WithLinearProbe model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dinov3 = DINOv3WithLinearProbe('dinov3_vitb16', freeze_backbone=False, num_layers_to_use=None)
    dinov3.to(device)
    summary(dinov3)
    
    input_tensor = torch.tensor(np.random.rand(32, 3, 256, 256), dtype=torch.float32).cuda()  # Example input tensor
    output = dinov3(input_tensor)  # Forward pass
    print("Output shape:", output.shape)  # Should print the shape of the output tensor