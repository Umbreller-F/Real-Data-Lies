from utils.experiment_utils import set_seed
# from data.dataset_split import GENVIDEO_PIKA, GENVIDEO_SEINE, MYVIDEOS, MYVIDEOS_COMPRESSED
# from data.video_dataset import get_video_dataset, get_composite_video_dataset
from omegaconf import DictConfig, OmegaConf
from models.timesformer import TimesformerBinaryClassifier
# from utils.train_utils import *
# from torch.utils.data import DataLoader
from loguru import logger
# from tqdm import tqdm
# from tabulate import tabulate
from PIL import Image

import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import hydra


@hydra.main(config_path="configs/experiments", config_name="standard-Pika-TALL.yaml", version_base=None)
def inference(cfg: DictConfig):
    # region Setup Logging
    logger.info(OmegaConf.to_yaml(cfg))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed)
    # endregion

    # region Load Model
    logger.info(f"Loading model: {cfg.model.name}")
    if 'timesformer' in cfg.model.name:
        model = TimesformerBinaryClassifier(model_name=cfg.model.name, pretrained=cfg.model.pretrained, freeze_backbone=False)
    else:
        raise NotImplementedError(f"Model {cfg.model.name} is not supported.")
    # endregion

    # region Load checkpoint
    ckpt_path = cfg.ckpt_path
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found at {ckpt_path}")
    logger.info(f"Loading checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    if torch.cuda.device_count() >= cfg.trainer.num_gpus and cfg.trainer.num_gpus > 1:
        model = nn.DataParallel(model, device_ids=cfg.trainer.device_ids[:cfg.trainer.num_gpus])
        model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    # endregion

    # region Load Demo Video
    video_dir = cfg.inference.demo_path
    label = cfg.inference.label
    processor = model.processor
    frame_paths = sorted(
        [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith('.jpg')],
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0].replace('frame', ''))
    )
    video_data = []
    for frame_path in frame_paths:
        img = Image.open(frame_path).convert('RGB')
        video_data.append(np.array(img))
    
    video = processor(images=video_data, return_tensors="pt").pixel_values[0]
    label = np.array([0 if label=="real" else 1], dtype=np.float32)
    video_id = os.path.basename(video_dir)
    # endregion

    # region Inference
    with torch.no_grad():
        video = video.unsqueeze(0).to(device)  # add batch dimension
        logits, attentions = model(video, output_attentions=True)
        output_pred = logits[:,0].sigmoid().cpu()
        predicted = output_pred > 0.5
    is_correct = (predicted.item() == label[0])
    correct_text = "CORRECT" if is_correct else "WRONG"
    logger.debug(f'Video: {video_id}, Label: {label}, Predicted: {predicted.item()}, Score: {output_pred.item():.4f}, Result: {correct_text}')
    # endregion

    draw_attention_map(attentions=attentions, video_data=video_data, layer=0, video_id=video_id, save_path='./results/attention_maps/')


def draw_attention_map(attentions, video_data, layer, video_id, save_path=None):
    video_attention = attentions[layer]
    # print(f"Attention shape at layer {layer}: {video_attention.shape}")
    for frame_idx in range(video_attention.shape[0]):
        frame_attention = video_attention[frame_idx, :, :, :]
        attention_map = frame_attention.mean(dim=0) # avg over heads
        cls_attention = attention_map[0, 1:] # get [CLS] token attention, shape: [196]
        spatial_attention = cls_attention.reshape(14, 14).cpu().numpy()
        original_frame = video_data[frame_idx]

        fig, axes = plt.subplots(1, 3, figsize=(18, 8))
        axes[0].imshow(original_frame)
        axes[0].set_title('Original Frame')
        axes[0].axis('off')
        im1 = axes[1].imshow(spatial_attention, cmap='viridis')
        axes[1].set_title('Attention Heatmap')
        axes[1].set_xlabel('Patch X (14x14 grid)')
        axes[1].set_ylabel('Patch Y')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        h, w = original_frame.shape[:2]
        attention_resized = np.array(Image.fromarray(spatial_attention).resize((w, h), Image.BILINEAR))
        overlay_alpha = 0.6
        overlay = original_frame.astype(np.float32) / 255.0
        attention_normalized = (attention_resized - attention_resized.min()) / (attention_resized.max() - attention_resized.min())
        heatmap = plt.cm.viridis(attention_normalized)[:, :, :3]
        combined = overlay * (1 - overlay_alpha) + heatmap * overlay_alpha
        axes[2].imshow(combined)
        axes[2].set_title('Attention Overlay')
        axes[2].axis('off')

        plt.tight_layout()
    
        if save_path:
            plt.savefig(save_path+f'{video_id}_{frame_idx}.png', dpi=300, bbox_inches='tight')

    images = [Image.open(save_path+f'{video_id}_{frame_idx}.png') for frame_idx in range(video_attention.shape[0])]
    
    widths, heights = zip(*(img.size for img in images))
    max_width = max(widths)
    total_height = sum(heights)
    
    combined_img = Image.new('RGB', (max_width, total_height))
    
    y_offset = 0
    for i, img in enumerate(images):
        combined_img.paste(img, (0, y_offset))
        y_offset += img.height
    
    combined_img.save(save_path+f'{video_id}_all.png', quality=95)


if __name__ == "__main__":
    inference()