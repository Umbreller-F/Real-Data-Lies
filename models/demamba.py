from transformers import XCLIPVisionModel, XCLIPProcessor
import os
import sys
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from models.mamba_base import MambaConfig, ResidualBlock
import torch.nn.init as init
from clip import clip
import math
from copy import deepcopy
from models.grl import GradientReverseLayer
from PIL import Image


def create_reorder_index(N, device):
    new_order = []
    for col in range(N):
        if col % 2 == 0:
            new_order.extend(range(col, N*N, N))
        else:
            new_order.extend(range(col + N*(N-1), col-1, -N))
    return torch.tensor(new_order, device=device)

def reorder_data(data, N):
    assert isinstance(data, torch.Tensor), "data should be a torch.Tensor"
    device = data.device
    new_order = create_reorder_index(N, device)
    B, t, _, _ = data.shape
    index = new_order.repeat(B, t, 1).unsqueeze(-1)
    reordered_data = torch.gather(data, 2, index.expand_as(data))
    return reordered_data

class XCLIP_DeMamba(nn.Module):
    def __init__(
        self, channel_size=768, class_num=1, quality_grl=False
    ):
        super(XCLIP_DeMamba, self).__init__()
        self.encoder = XCLIPVisionModel.from_pretrained("microsoft/xclip-base-patch16", local_files_only=True)
        blocks = []
        channel = 768
        self.fusing_ratios = 1
        self.patch_nums = (14//self.fusing_ratios)**2
        self.mamba_configs = MambaConfig(d_model=channel)
        self.mamba = ResidualBlock(config = self.mamba_configs)
        self.fc1 = nn.Linear((self.patch_nums+1)*channel, class_num)
        self.fc_norm = nn.LayerNorm(self.patch_nums*channel)
        self.fc_norm2 = nn.LayerNorm(768)
        self.initialize_weights(self.fc1)
        self.dropout = nn.Dropout(p=0.0)
        self._processor = XCLIPProcessor.from_pretrained("microsoft/xclip-base-patch16", local_files_only=True).image_processor
        #debug
        # self._processor = None

        self.quality_grl = quality_grl
        if self.quality_grl:
            self.grl = GradientReverseLayer()
            # self.score_fc = nn.Linear((self.patch_nums+1)*channel, 1)
            self.score_fc = nn.Sequential(
                nn.Linear((self.patch_nums+1)*channel, channel),
                nn.ReLU(),
                nn.Linear(channel, 1),
            )


    def initialize_weights(self, module):
        for m in module.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, x, alpha=1.0):
        b, t, _, h, w = x.shape
        images = x.view(b * t, 3, h, w)
        outputs = self.encoder(images, output_hidden_states=True)
        sequence_output = outputs['last_hidden_state'][:,1:,:]
        _, _, c = sequence_output.shape

        global_feat = outputs['pooler_output'].reshape(b, t, -1)
        global_feat = global_feat.mean(1)
        global_feat = self.fc_norm2(global_feat)

        sequence_output = sequence_output.view(b, t, -1, c)
        _, _, f_w, _ = sequence_output.shape
        f_h, f_w = int(math.sqrt(f_w)), int(math.sqrt(f_w))

        s = f_h//self.fusing_ratios
        sequence_output = sequence_output.view(b, t, self.fusing_ratios, s, self.fusing_ratios, s, c)
        x = sequence_output.permute(0, 2, 4, 1, 3, 5, 6).contiguous().view(b*s*s, t, -1, c)
        b_l = b*s*s
        
        x = reorder_data(x, self.fusing_ratios)
        x = x.permute(0, 2, 1, 3).contiguous().view(b_l, -1, c)
        res = self.mamba(x)

        video_level_features = res.mean(1)
        video_level_features = video_level_features.view(b, -1)
        video_level_features = self.fc_norm(video_level_features)
        video_level_features = torch.cat((global_feat, video_level_features), dim=1)

        pred = self.fc1(video_level_features)
        pred = self.dropout(pred)

        if self.quality_grl:
            feat_grl = self.grl(video_level_features, alpha)
            pred_score_raw = self.score_fc(feat_grl)
            pred_score = torch.sigmoid(pred_score_raw)
            return pred, pred_score

        return pred

    @property
    def processor(self):
        return deepcopy(self._processor)


class XCLIP_DeMamba_Q1(nn.Module):
    def __init__(
        self, channel_size=768, class_num=1, 
    ):
        super(XCLIP_DeMamba_Q1, self).__init__()
        self.encoder = XCLIPVisionModel.from_pretrained("microsoft/xclip-base-patch16", local_files_only=True)
        blocks = []
        channel = 768
        self.fusing_ratios = 1
        self.patch_nums = (14//self.fusing_ratios)**2
        self.mamba_configs = MambaConfig(d_model=channel)
        self.mamba = ResidualBlock(config = self.mamba_configs)

        self.attr_embed_dim = channel  # 768
        self.attr_mlp = nn.Sequential(
            nn.Linear(3, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, self.attr_embed_dim),
            nn.ReLU()
        )
        total_input_dim = (self.patch_nums + 1) * channel + self.attr_embed_dim

        self.fc1 = nn.Linear(total_input_dim, class_num)
        self.fc_norm = nn.LayerNorm(self.patch_nums*channel)
        self.fc_norm2 = nn.LayerNorm(768)
        self.initialize_weights(self.fc1)
        self.dropout = nn.Dropout(p=0.0)
        self._processor = XCLIPProcessor.from_pretrained("microsoft/xclip-base-patch16", local_files_only=True).image_processor


    def initialize_weights(self, module):
        for m in module.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, x, attributes):
        b, t, _, h, w = x.shape
        images = x.view(b * t, 3, h, w)
        outputs = self.encoder(images, output_hidden_states=True)
        sequence_output = outputs['last_hidden_state'][:,1:,:]
        _, _, c = sequence_output.shape

        global_feat = outputs['pooler_output'].reshape(b, t, -1)
        global_feat = global_feat.mean(1)
        global_feat = self.fc_norm2(global_feat)

        sequence_output = sequence_output.view(b, t, -1, c)
        _, _, f_w, _ = sequence_output.shape
        f_h, f_w = int(math.sqrt(f_w)), int(math.sqrt(f_w))

        s = f_h//self.fusing_ratios
        sequence_output = sequence_output.view(b, t, self.fusing_ratios, s, self.fusing_ratios, s, c)
        x = sequence_output.permute(0, 2, 4, 1, 3, 5, 6).contiguous().view(b*s*s, t, -1, c)
        b_l = b*s*s
        
        x = reorder_data(x, self.fusing_ratios)
        x = x.permute(0, 2, 1, 3).contiguous().view(b_l, -1, c)
        res = self.mamba(x)

        video_level_features = res.mean(1)
        video_level_features = video_level_features.view(b, -1)
        video_level_features = self.fc_norm(video_level_features)

        # attributes shape: [B, 3] -> [B, 768]
        attr_feat = self.attr_mlp(attributes) 
        
        # [Global Feature, Mamba Feature, Attribute Feature]
        combined_features = torch.cat((global_feat, video_level_features, attr_feat), dim=1)
        # video_level_features = torch.cat((global_feat, video_level_features), dim=1)

        pred = self.fc1(combined_features)
        pred = self.dropout(pred)

        return pred

    @property
    def processor(self):
        return deepcopy(self._processor)


class XCLIP_DeMamba_Q2(nn.Module):
    def __init__(self, channel_size=768, class_num=1):
        super(XCLIP_DeMamba_Q2, self).__init__()
        self.encoder = XCLIPVisionModel.from_pretrained("microsoft/xclip-base-patch16", local_files_only=True)
        
        channel = 768
        self.fusing_ratios = 1
        self.patch_nums = (14//self.fusing_ratios)**2
        
        self.mamba_configs = MambaConfig(d_model=channel)
        self.mamba = ResidualBlock(config=self.mamba_configs)
        
        # -----------------------------------------------------
        self.attr_embed_dim = channel 
        self.attr_mlp = nn.Sequential(
            nn.Linear(3, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, self.attr_embed_dim),
            nn.LayerNorm(self.attr_embed_dim)
        )
        # -----------------------------------------------------

        self.fc_norm = nn.LayerNorm(self.patch_nums*channel)
        self.fc_norm2 = nn.LayerNorm(768)
        self.fc1 = nn.Linear((self.patch_nums+1)*channel, class_num)
        
        self.initialize_weights(self.fc1)
        self.initialize_weights(self.attr_mlp)
        
        self.dropout = nn.Dropout(p=0.0)
        self._processor = XCLIPProcessor.from_pretrained("microsoft/xclip-base-patch16", local_files_only=True).image_processor

    def initialize_weights(self, module):
        for m in module.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, x, attributes):
        """
        x: [B, T, C, H, W]
        attributes: [B, 3]
        """
        b, t, _, h, w = x.shape
        
        # attr_token: [B, 768]
        attr_token = self.attr_mlp(attributes) 

        # --- XCLIP Encoder ---
        images = x.view(b * t, 3, h, w)
        outputs = self.encoder(images, output_hidden_states=True)
        sequence_output = outputs['last_hidden_state'][:,1:,:]
        _, _, c = sequence_output.shape

        global_feat = outputs['pooler_output'].reshape(b, t, -1)
        global_feat = global_feat.mean(1)
        global_feat = self.fc_norm2(global_feat)

        # --- Reshape & Reorder ---
        sequence_output = sequence_output.view(b, t, -1, c)
        _, _, f_w, _ = sequence_output.shape
        f_h, f_w = int(math.sqrt(f_w)), int(math.sqrt(f_w))

        s = f_h//self.fusing_ratios
        sequence_output = sequence_output.view(b, t, self.fusing_ratios, s, self.fusing_ratios, s, c)
        
        # [B*S*S, T, -1, C]
        x_reshaped = sequence_output.permute(0, 2, 4, 1, 3, 5, 6).contiguous().view(b*s*s, t, -1, c)
        b_l = b*s*s
        
        x_reshaped = reorder_data(x_reshaped, self.fusing_ratios)
        
        # [Batch_Large, Sequence_Length, Channel]
        x_mamba_input = x_reshaped.permute(0, 2, 1, 3).contiguous().view(b_l, -1, c)
        
        # ------------------------------------------------------------------
        num_patches = s * s
        # attr_token: [B, C] -> [B, 1, C] -> [B, num_patches, C]
        attr_expanded = attr_token.unsqueeze(1).repeat(1, num_patches, 1)
        # [B * num_patches, 1, C] [b_l, 1, C]
        attr_expanded = attr_expanded.view(b * num_patches, 1, c)

        # x_with_attr shape: [b_l, SeqLen + 1, C]
        x_with_attr = torch.cat([attr_expanded, x_mamba_input], dim=1) 
        # ------------------------------------------------------------------
        
        res = self.mamba(x_with_attr) 

        video_level_features = res.mean(1) 
        
        video_level_features = video_level_features.view(b, -1)
        video_level_features = self.fc_norm(video_level_features)
        
        final_features = torch.cat((global_feat, video_level_features), dim=1)

        pred = self.fc1(final_features)
        pred = self.dropout(pred)

        return pred

    @property
    def processor(self):
        return deepcopy(self._processor)


def custom_forward(self, x):
    x = self.conv1(x)  # shape = [*, width, grid, grid]
    x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
    x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
    x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
    x = x + self.positional_embedding.to(x.dtype)
    x = self.ln_pre(x)

    x = x.permute(1, 0, 2)  # NLD -> LND
    x = self.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD

    x = self.ln_post(x[:, 1:, :])

    if self.proj is not None:
        x = x @ self.proj

    return x


class VideoProcessorWrapper:
    def __init__(self, processor):
        self.processor = processor
    
    def __call__(self, images, return_tensors="pt"):
        processed_frames = []
        for frame in images:
            img = Image.fromarray(frame)
            tensor = self.processor(img)
            processed_frames.append(tensor)
        
        video_tensor = torch.stack(processed_frames)  # [T, C, H, W]
        
        if return_tensors == "pt":
            class Result:
                pixel_values = [video_tensor]
            return Result


class CLIP_DeMamba(nn.Module):
    def __init__(
        self, channel_size=512, class_num=1
    ):
        super(CLIP_DeMamba, self).__init__()
        self.clip_model, preprocess = clip.load('ViT-B/16')
        self.clip_model = self.clip_model.float()
        self.clip_model.visual.forward = custom_forward.__get__(self.clip_model.visual, type(self.clip_model.visual))
        blocks = []
        channel = 512
        self.fusing_ratios = 2
        self.patch_nums = (14//self.fusing_ratios)**2
        self.mamba_configs = MambaConfig(d_model=channel)
        self.mamba = ResidualBlock(config = self.mamba_configs)
        self.fc1 = nn.Linear(channel*(self.patch_nums+1), class_num)
        self.bn1 = nn.BatchNorm1d(channel)
        self.initialize_weights(self.fc1)

        self._processor = VideoProcessorWrapper(preprocess)

    def initialize_weights(self, module):
        for m in module.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, x):
        b, t, _, h, w = x.shape
        images = x.view(b * t, 3, h, w)
        sequence_output = self.clip_model.encode_image(images)
        _, _, c = sequence_output.shape
        sequence_output = sequence_output.view(b, t, -1, c)

        global_feat = sequence_output.reshape(b, -1, c)
        global_feat = global_feat.mean(1)

        _, _, f_w, _ = sequence_output.shape
        f_h, f_w = int(math.sqrt(f_w)), int(math.sqrt(f_w))

        s = f_h//self.fusing_ratios
        sequence_output = sequence_output.view(b, t, self.fusing_ratios, s, self.fusing_ratios, s, c)
        x = sequence_output.permute(0, 2, 4, 1, 3, 5, 6).contiguous().view(b*s*s, t, -1, c)
        b_l = b*s*s
        
        x = reorder_data(x, self.fusing_ratios)
        x = x.permute(0, 2, 1, 3).contiguous().view(b_l, -1, c)
        res = self.mamba(x)
        video_level_features = res.mean(1)
        video_level_features = video_level_features.view(b, -1)

        video_level_features = torch.cat((global_feat, video_level_features), dim=1)
        x = self.fc1(video_level_features)

        return x
    
    @property
    def processor(self):
        return deepcopy(self._processor)


if __name__ == '__main__':
    from torchinfo import summary
    # model = XCLIP_DeMamba()
    model = CLIP_DeMamba()
    summary(model)
    model = model.cuda()
    print(model.processor)
    # breakpoint()
    tensor = torch.tensor(np.random.rand(2, 8, 3, 224, 224), dtype=torch.float32).cuda()
    output = model(tensor)
    print(output.shape)

    # processor = XCLIPProcessor.from_pretrained("microsoft/xclip-base-patch16", local_files_only=True).image_processor
    # print(processor)
    # breakpoint()

    # model = XCLIP_DeMamba(quality_grl=True)
    # summary(model)
    # model = model.cuda()
    # tensor = torch.tensor(np.random.rand(2, 8, 3, 224, 224), dtype=torch.float32).cuda()
    # output = model(tensor)
    # breakpoint()

    # model = XCLIP_DeMamba_Q1()
    # summary(model)
    # model = model.cuda()
    # model.eval()
    # video_tensor = torch.tensor(np.random.rand(2, 8, 3, 224, 224), dtype=torch.float32).cuda()
    # Q_attributes = torch.tensor(np.random.rand(2, 3) * 100, dtype=torch.float32).cuda()  # Example attributes
    # output = model(video_tensor, Q_attributes)
    # print(output.shape)

    # model = XCLIP_DeMamba_Q2()
    # summary(model)
    # model = model.cuda()
    # model.eval()
    # video_tensor = torch.tensor(np.random.rand(2, 8, 3, 224, 224), dtype=torch.float32).cuda()
    # Q_attributes = torch.tensor(np.random.rand(2, 3) * 100, dtype=torch.float32).cuda()  # Example attributes
    # output = model(video_tensor, Q_attributes)
    # print(output.shape)