import os
import random
import warnings
import logging
import numpy as np
import pandas as pd
import albumentations as A

from PIL import Image, ImageOps
from torch.utils.data import Dataset, ConcatDataset
from data.utils import *
from loguru import logger
from torchvision import transforms
from vidaug import augmentors as va
from albumentations.core.composition import Compose, ReplayCompose, OneOf
from models.dct import DCT_base_Rec_Module


if not hasattr(np, 'float'):
    np.float = float

warnings.filterwarnings("ignore", message="Creating a tensor from a list of numpy.ndarrays is extremely slow")
warnings.filterwarnings("ignore", message="`resume_download` is deprecated", category=FutureWarning)


class ImageDataset(Dataset):
    def __init__(self, processor, data_path:str, dataset_name:str="GenVideo", generation_model:str="None", frame_sample_rate: int = 4,
                 mode: str = "train", num_frames: int = 8, load_len: int = None, input_shape: tuple = (224, 224),):
        super().__init__()
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.generation_model = generation_model
        self.mode = mode
        self.processor = processor
        self.num_frames = num_frames
        self.frame_sample_rate = frame_sample_rate
        self.input_shape = input_shape
        self.transform = transforms.Compose([
                            # transforms.Resize(input_shape),
                            transforms.Resize(input_shape[0], interpolation=transforms.InterpolationMode.BILINEAR),
                            transforms.CenterCrop(input_shape),
                            transforms.ToTensor(),
                            transforms.Normalize(
                            mean=[0.485, 0.456, 0.406],    # Mean of ImageNet
                            std=[0.229, 0.224, 0.225]),    # Std of ImageNet
                            ])
        # AIDE transform
        self.transform_before = transforms.ToTensor()
        self.dct = DCT_base_Rec_Module()
        self.transform_after = transforms.Compose([
                transforms.Resize([256, 256]),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        self.label = get_label_from_generation_model(self.generation_model)
        # data_dir
        self.base_dir = os.path.join(
            self.data_path, 'video_frames', self.label, self.generation_model, self.mode
        )
        # all videos path
        self.video_dirs = sorted(
            [os.path.join(self.base_dir, d) for d in os.listdir(self.base_dir)],
            key=lambda x: os.path.basename(x)  # sort by video number
        )
        # limit dataset size
        if load_len is not None:
            # logger.info(f"Limiting dataset from {len(self.video_dirs)} to {load_len} videos.")
            self.video_dirs = self.video_dirs[:load_len]
        # all video frames path
        self.image_paths = []
        required_total_frames = (self.num_frames - 1) * self.frame_sample_rate + 1
        for video_dir in self.video_dirs:
            # Check if the directory exists and contains .jpg files
            if os.path.isdir(video_dir):
                frames = sorted(
                    [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(('.jpg', '.png'))],
                    key=lambda x: int(os.path.splitext(os.path.basename(x))[0].replace('frame', ''))
                )
                if len(frames) < self.num_frames:
                    continue
                elif len(frames) >= required_total_frames:
                    sampled_frames = frames[:required_total_frames:self.frame_sample_rate]
                else:
                    max_sample_rate = max(1, (len(frames) - 1) // (self.num_frames - 1))
                    sampled_frames = frames[::max_sample_rate][:self.num_frames]
                self.image_paths.extend(sampled_frames)
        logger.success(f"[{self.dataset_name} / {self.mode} / {len(self)} images / {len(self.video_dirs)} videos / {self.generation_model}]")
            
    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> np.ndarray:
        image_path = self.image_paths[idx]

        img = Image.open(image_path).convert('RGB')
        
        if self.processor is None:
            img = self.transform(img)
        elif self.processor == 'AIDE-specific':
            feature_path = image_path.replace('.jpg', '.pt').replace('video_frames', 'dct_features')
            dct_features = torch.load(feature_path, map_location='cpu')
            x0 = self.transform_after(self.transform_before(img))
            img = []
            for key in ['x_minmin', 'x_maxmax', 'x_minmin1', 'x_maxmax1']:
                img.append(self.transform_after(dct_features[key]))
            img.append(x0)
            img = torch.stack(img, dim=0)
        else:
            img = self.processor(images=img, return_tensors="pt").pixel_values[0]
        
        label = np.array([0 if self.label=="real" else 1], dtype=np.float32)
        img_id = '/'.join(image_path.split('/')[-2:])
        return img, label, img_id


class VideoDataset(Dataset):
    def __init__(self, processor, generation_model: str, data_path: str, sample_strategy='fixed_interval', frame_sample_rate: int = 4,
                 mode: str = "train", load_len: int = None, num_frames: int = 8, dataset_name: str = "GenVideo", input_shape = (224, 224),):
        super().__init__()
        assert num_frames > 1, f"num_frames must be greater than 1, but got {num_frames}"
        self.num_frames = num_frames
        SUPPORTED_STRATEGIES = ['consecutive', 'fixed_interval'] # ['uniform', 'consecutive', 'fixed_interval']
        assert sample_strategy in SUPPORTED_STRATEGIES, f"Sample strategy {sample_strategy} is not supported."
        self.sample_strategy = sample_strategy
        if self.sample_strategy == "fixed_interval":
            logger.info(f'Fixed frame sampling rate: 1/{frame_sample_rate}, {self.num_frames} frames per sample.')
        elif self.sample_strategy == "consecutive":
            self.frame_sample_rate = 1
            logger.info(f'Use consecutive frame sampling, {self.num_frames} frames per sample.')
        elif self.sample_strategy == 'uniform':
            # Not implemented yet
            logger.info(f'Use uniform sampling, {self.num_frames} frames per sample.')
        self.mode = mode
        self.processor = processor
        logging.getLogger("transformers").setLevel(logging.ERROR)
        if self.processor is not None:
            logger.info("Use backbone's standard preprocessing pipeline.")
        else:
            self.transform = transforms.Compose([
                                # transforms.Resize(input_shape),
                                transforms.Resize(input_shape[0], interpolation=transforms.InterpolationMode.BILINEAR),
                                transforms.CenterCrop(input_shape),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                mean=[0.485, 0.456, 0.406],  # Mean of ImageNet
                                std=[0.229, 0.224, 0.225]),    # Std of ImageNet
                                ])
            logger.info(f"No processor passed in, use general transform for preprocessing:\n{self.transform}")

        self.input_shape = input_shape
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.generation_model = generation_model
        self.label = get_label_from_generation_model(self.generation_model)
        # data_dir
        frames_dir = 'video_frames'
        self.frame_sample_rate = frame_sample_rate
        self.base_dir = os.path.join(
            self.data_path, frames_dir, self.label, self.generation_model, self.mode
        )
        # all videos path
        self.video_dirs = sorted(
            [os.path.join(self.base_dir, d) for d in os.listdir(self.base_dir)],
            key=lambda x: os.path.basename(x)  # sort by video number
        )
        # all video frames path
        self.video_frame_paths = []
        required_total_frames = (self.num_frames - 1) * self.frame_sample_rate + 1
        for video_dir in self.video_dirs:
            # Check if the directory exists and contains .jpg files
            if os.path.isdir(video_dir):
                frames = sorted(
                    [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(('.jpg', '.png'))],
                    key=lambda x: int(os.path.splitext(os.path.basename(x))[0].replace('frame', ''))
                )
                if len(frames) < self.num_frames:
                    continue
                elif len(frames) >= required_total_frames:
                    sampled_frames = frames[:required_total_frames:self.frame_sample_rate]
                else:
                    max_sample_rate = max(1, (len(frames) - 1) // (self.num_frames - 1))
                    sampled_frames = frames[::max_sample_rate][:self.num_frames]
                self.video_frame_paths.append(sampled_frames)
        
        if len(self.video_frame_paths) == 0:
            logger.warning(f"No valid videos found in {self.base_dir} with at least {self.num_frames} frames.")
            return
        
        if load_len is not None:
            if len(self.video_frame_paths) > load_len:
                logger.info(f"Limiting dataset from {len(self.video_frame_paths)} to {load_len} videos.")
                self.video_frame_paths = self.video_frame_paths[:load_len]
        logger.success(f"[{self.dataset_name} / {self.mode} / {len(self)} videos / {self.generation_model}]")
    
    def __len__(self) -> int:
        return len(self.video_frame_paths)
    
    def __getitem__(self, idx):
        frame_paths = self.video_frame_paths[idx]
        if self.processor is None:
            video_data = []
            for frame_path in frame_paths:
                img = Image.open(frame_path).convert('RGB')
                
                if self.transform:
                    img = self.transform(img)
                
                # convert (H, W, C) to (C, H, W)
                img_array = np.array(img)
                video_data.append(img_array)
            # merge frames to video
            video = np.stack(video_data, axis=0)
            label = np.array([0 if self.label=="real" else 1], dtype=np.float32)
            video_id = frame_path.split('/')[-2]
            return video, label, video_id
        else:
            video_data = []
            for frame_path in frame_paths:
                img = Image.open(frame_path).convert('RGB')
                video_data.append(img)
            
            video_data = [np.array(img) for img in video_data]
            
            video = self.processor(images=video_data, return_tensors="pt").pixel_values[0]
            label = np.array([0 if self.label=="real" else 1], dtype=np.float32)
            video_id = frame_path.split('/')[-2]
            
            return video, label, video_id


def get_paired_dataset(data_cfg, mode, processor, load_len=None, num_frames=8, frame_sample_rate=4,
                       generation_model=None, real_model=None, pn_ratio=1, sample_strategy='fixed_interval'):
    """
    Load and concatenate video datasets for fake and real videos.
    
    Args:
        data_cfg: Configuration object containing dataset parameters
        mode (str): Dataset mode - 'train', 'val', or 'test'
        processor: Video processor for data preprocessing
        load_len (int, optional): Total number of samples to load. If None, loads all available data
        generation_model (str, optional): Model name for generated/fake videos
        real_model (str, optional): Model name for real videos  
        pn_ratio (float, optional): Positive-negative ratio for balancing real vs fake samples. 
                                   Defaults to 1 (equal ratio)
    
    Returns:
        ConcatDataset: Concatenated dataset containing both fake and real video samples
        
    Example:
        >>> dataset = get_video_dataset(cfg, 'train', processor, load_len=1000, pn_ratio=2)
        >>> # This will load 1000 total samples with 2:1 fake-to-real ratio
    """
    feature_type = data_cfg.feature_type
    logger.info(f"Using feature type : {feature_type.upper()}")
    if feature_type == "video":
        fake_dataset = VideoDataset(
            processor=processor,
            data_path=data_cfg.data_path, 
            dataset_name=data_cfg.dataset_name,
            generation_model=generation_model,
            sample_strategy=sample_strategy,
            frame_sample_rate=frame_sample_rate,
            mode=mode, 
            num_frames=num_frames,
            load_len=load_len,
            input_shape=tuple(data_cfg.input_shape),
            )
        real_len = int(len(fake_dataset) / pn_ratio)
        real_dataset = VideoDataset(
            processor=processor,
            data_path=data_cfg.data_path, 
            dataset_name=data_cfg.dataset_name,
            generation_model=real_model,
            sample_strategy=sample_strategy,
            frame_sample_rate=frame_sample_rate,
            mode=mode, 
            num_frames=num_frames,
            load_len=real_len,
            input_shape=tuple(data_cfg.input_shape),
            )
    elif feature_type == "image":
        fake_dataset = ImageDataset(
            processor=processor,
            data_path=data_cfg.data_path, 
            dataset_name=data_cfg.dataset_name,
            generation_model=generation_model,
            frame_sample_rate=frame_sample_rate,
            mode=mode, 
            num_frames=num_frames,
            load_len=load_len,
            input_shape=tuple(data_cfg.input_shape),
            )
        real_len = int(len(fake_dataset) / pn_ratio / num_frames)
        real_dataset = ImageDataset(
            processor=processor,
            data_path=data_cfg.data_path, 
            dataset_name=data_cfg.dataset_name,
            generation_model=real_model,
            frame_sample_rate=frame_sample_rate,
            mode=mode, 
            num_frames=num_frames,
            load_len=real_len,
            input_shape=tuple(data_cfg.input_shape),
            )
    else:
        raise NotImplementedError(f"Feature type {feature_type} is not supported.")
    # breakpoint()
    return ConcatDataset([fake_dataset, real_dataset])


def get_single_dataset(data_cfg, mode, processor, load_len=None, num_frames=8, frame_sample_rate=4,
                       data_model=None, sample_strategy='fixed_interval'):
    feature_type = data_cfg.feature_type
    logger.info(f"Using feature type : {feature_type.upper()}")
    if feature_type == "video":
        dataset = VideoDataset(
            processor=processor,
            data_path=data_cfg.data_path, 
            dataset_name=data_cfg.dataset_name,
            generation_model=data_model,
            sample_strategy=sample_strategy,
            frame_sample_rate=frame_sample_rate,
            mode=mode, 
            num_frames=num_frames,
            load_len=load_len,
            input_shape=tuple(data_cfg.input_shape),
        )
    elif feature_type == "image":
        dataset = ImageDataset(
            processor=processor,
            data_path=data_cfg.data_path, 
            dataset_name=data_cfg.dataset_name,
            generation_model=data_model,
            frame_sample_rate=frame_sample_rate,
            mode=mode, 
            num_frames=num_frames,
            load_len=load_len,
            input_shape=tuple(data_cfg.input_shape),
        )
    else:
        raise NotImplementedError(f"Feature type {feature_type} is not supported.")
    return dataset


if __name__ == "__main__":
    # use python -m data.dataset to test
    logger.debug("This module is not meant to be run directly. Import it in your code to use the functions and classes defined here.")
    from omegaconf import OmegaConf
    from transformers import AutoImageProcessor
    from torch.utils.data import DataLoader
    from copy import deepcopy
    from utils.experiment_utils import set_seed, seed_worker
    set_seed(1)
    generator = torch.Generator()
    generator.manual_seed(1958)
    
    pn_ratio = 1
    cfg = OmegaConf.create({
        'data': {
            'batch_size': 24,
            'val_batch_size': 8,
            'feature_type': 'video',
            'dataset_name': "GenVideo",
            'num_workers': 16,
            'data_path': "../Data/GenVideo",
            'num_frames': 8,
            'generation_model': "Pika",
            'input_shape': [224,224],
            'train_real_model': "Kinetics-400",
            'val_real_model': "Kinetics-400",
            'test_real_model': "MSR-VTT",
            'train_load_len': 10000,
            'val_load_len': 100,
            'diffuse_steps': 5,
            'sample_strategy': 'fixed_interval',
        }
    })
    
    logger.debug("Testing VideoDataset...")
    demo_dataset = VideoDataset(processor=AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-ssv2", use_fast=False, local_files_only=True), 
                                data_path="../Data/RealDist", dataset_name="RealDist",  generation_model="Uniform", mode="train")
    # demo_loader = DataLoader(demo_dataset, batch_size=cfg.data.batch_size, shuffle=True, num_workers=cfg.data.num_workers,
    #                          worker_init_fn=seed_worker, generator=generator)
    # for idx, batch in enumerate(demo_loader):
    #     videos, labels, video_ids = batch
    #     print(video_ids)
    #     break
    # logger.debug("Testing get_video_dataset...")
    # processsor = AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-ssv2", use_fast=False, local_files_only=True)
    # real_fake_dataset = get_video_dataset(cfg.data, generation_model=cfg.data.generation_model, 
    #                                       real_model=cfg.data.train_real_model, mode="train", 
    #                                       load_len=cfg.data.train_load_len, pn_ratio=pn_ratio,
    #                                       processor=deepcopy(processsor), no_resize=cfg.data.no_resize)
    # print(processsor)
    # logger.debug(f"Total training samples: {len(real_fake_dataset)}")

    # demo_loader = DataLoader(real_fake_dataset, batch_size=cfg.data.batch_size, shuffle=True, num_workers=cfg.data.num_workers)
    # for batch in demo_loader:
    #     videos, labels, video_ids = batch
    #     break

    logger.debug("Testing ImageDataset...")
    imagedataset_demo = ImageDataset(processor=AutoImageProcessor.from_pretrained('facebook/dinov2-base', local_files_only=True),
        data_path=cfg.data.data_path, dataset_name=cfg.data.dataset_name, input_shape=tuple(cfg.data.input_shape), generation_model="Pika", mode="train")
    demo_loader = DataLoader(imagedataset_demo, batch_size=cfg.data.batch_size, shuffle=False, num_workers=cfg.data.num_workers)
    for batch in demo_loader:
        images, labels, image_ids = batch
        print(images.shape)
        print(image_ids)
        break
    breakpoint()