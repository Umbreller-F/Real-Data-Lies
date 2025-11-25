from pathlib import Path
from loguru import logger

import torch
import os

from data.hunyuan_vae.autoencoder_kl_causal_3d import AutoencoderKLCausal3D

MODEL_BASE = os.getenv("MODEL_BASE", "./ckpts")
VAE_PATH = {"884-16c-hy": f"{MODEL_BASE}/hunyuan-video-t2v-720p/vae"}
PRECISION_TO_TYPE = {
    'fp32': torch.float32,
    'fp16': torch.float16,
    'bf16': torch.bfloat16,
}

def load_vae(vae_type: str="884-16c-hy",
             vae_precision: str='fp16',
             sample_size: tuple=None,
             vae_path: str=None,
             device='cuda'
             ):
    """the fucntion to load the 3D VAE model

    Args:
        vae_type (str): the type of the 3D VAE model. Defaults to "884-16c-hy".
        vae_precision (str, optional): the precision to load vae. Defaults to None.
        sample_size (tuple, optional): the tiling size. Defaults to None.
        vae_path (str, optional): the path to vae. Defaults to None.
        device (_type_, optional): device to load vae. Defaults to None.
    """
    if vae_path is None:
        vae_path = VAE_PATH[vae_type]
    
    logger.info(f"Loading 3D VAE model ({vae_type}) from: {vae_path}")
    config = AutoencoderKLCausal3D.load_config(vae_path)
    if sample_size:
        vae = AutoencoderKLCausal3D.from_config(config, sample_size=sample_size)
    else:
        vae = AutoencoderKLCausal3D.from_config(config)
    
    vae_ckpt = Path(vae_path) / "pytorch_model.pt"
    assert vae_ckpt.exists(), f"VAE checkpoint not found: {vae_ckpt}"
    
    ckpt = torch.load(vae_ckpt, map_location=vae.device)
    if "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    if any(k.startswith("vae.") for k in ckpt.keys()):
        ckpt = {k.replace("vae.", ""): v for k, v in ckpt.items() if k.startswith("vae.")}
    vae.load_state_dict(ckpt)

    spatial_compression_ratio = vae.config.spatial_compression_ratio
    time_compression_ratio = vae.config.time_compression_ratio
    
    if vae_precision is not None:
        vae = vae.to(dtype=PRECISION_TO_TYPE[vae_precision])

    vae.requires_grad_(False)

    logger.info(f"VAE to dtype: {vae.dtype}")

    if device is not None:
        vae = vae.to(device)

    vae.eval()

    return vae
