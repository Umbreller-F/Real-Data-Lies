import torch
from transformers import AutoConfig
from transformers import VideoMAEImageProcessor
from models.VideoMAEv2_Base.modeling_videomaev2 import VideoMAEv2, Block
from copy import deepcopy
import numpy as np
import torch.nn as nn
from functools import partial

class VideoMAEv2_X(nn.Module):
    def __init__(self, dropout_rate: float = 0.1, output_dim: int = 1):
        super().__init__()
        config = AutoConfig.from_pretrained("models/VideoMAEv2_Base", trust_remote_code=True, local_files_only=True)
        self.pretrained_backbone = VideoMAEv2.from_pretrained(
            "models/VideoMAEv2_Base",
            config=config,
            use_safetensors=True,
            local_files_only=True
        ).model
        self._processor = VideoMAEImageProcessor.from_pretrained(f"OpenGVLab/VideoMAEv2-Base", local_files_only=True)
        # breakpoint()
        self.extra_blocks = nn.ModuleList([
            Block(
                dim=config.model_config['embed_dim'],
                num_heads=config.model_config['num_heads'],
                mlp_ratio=config.model_config['mlp_ratio'],
                qkv_bias=config.model_config['qkv_bias'],
                qk_scale=config.model_config['qk_scale'],
                drop=config.model_config['drop_rate'],
                attn_drop=config.model_config['attn_drop_rate'],
                drop_path=0.0,
                norm_layer=partial(eval(config.model_config['norm_layer']), eps=config.model_config['layer_norm_eps']),
                init_values=config.model_config['init_values'],
                cos_attn=config.model_config['cos_attn']) for _ in range(1)
        ])
        self.fc1 = nn.Linear(150528+768, output_dim)
        self.fc_norm = nn.LayerNorm(150528)
        self.fc_norm2 = nn.LayerNorm(768)
        # self.classifier = nn.Sequential(
        #     nn.Dropout(dropout_rate),
        #     nn.Linear(768, 512),
        #     nn.ReLU(),
        #     nn.Dropout(dropout_rate),
        #     nn.Linear(512, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, output_dim)
        # )

    def forward(self, x):
        inputs = x.permute(0, 2, 1, 3, 4)
        b, c, t, h, w = inputs.shape
        tokens = self.pretrained_backbone.get_all_tokens(inputs)
        global_feat = tokens.reshape(b, -1, 768).mean(1)
        global_feat = self.fc_norm2(global_feat)
        for blk in self.extra_blocks:
            tokens = blk(tokens)
        tokens = tokens.reshape(b, 8, -1)
        video_level_features = tokens.mean(1)
        video_level_features = self.fc_norm(video_level_features)
        video_level_features = torch.cat((global_feat, video_level_features), dim=1)
        # logits = self.classifier(video_level_features)
        logits = self.fc1(video_level_features)
        return logits

    @property
    def processor(self):
        return deepcopy(self._processor)


if __name__ == "__main__":
    from torchinfo import summary
    '''# 1. 直接从你的本地文件导入模型类
    # 假设 modeling_videomae_v2.py 里定义的类名是 VideoMAEv2Model

    model_path = "models/VideoMAEv2_Base"  # 包含 safetensors 和 config 的文件夹路径

    # 2. 加载配置
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)

    # 3. 直接使用该类加载预训练权重
    # transformers 会自动检测文件夹下的 model.safetensors
    model = VideoMAEv2.from_pretrained(
        model_path,
        config=config,
        use_safetensors=True,  # 显式指定使用 safetensors
        local_files_only=True
    )

    print("模型加载成功！")
    breakpoint()

    processor = VideoMAEImageProcessor.from_pretrained(f"OpenGVLab/VideoMAEv2-Base", local_files_only=True)
    # video = list(np.random.rand(16, 3, 224, 224))
    video = [np.random.randint(0, 256, (3, 224, 224), dtype=np.uint8) for _ in range(16)]

    # B, T, C, H, W -> B, C, T, H, W
    inputs = processor(video, return_tensors="pt")
    inputs['pixel_values'] = inputs['pixel_values'].permute(0, 2, 1, 3, 4)

    with torch.no_grad():
        outputs = model(**inputs)
    print("Logits shape:", outputs.shape)'''
    
    model = VideoMAEv2_X()
    summary(model)
    print(model.processor)
    model = model.cuda()
    tensor = torch.tensor(np.random.rand(4, 16, 3, 224, 224), dtype=torch.float32).cuda()
    with torch.no_grad():
        logits = model(tensor)
    print("Logits shape:", logits.shape)
    breakpoint()