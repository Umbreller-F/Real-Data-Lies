import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset
from typing import Optional, Callable
from data.utils import *
from pprint import pprint as print
from loguru import logger
from torchvision import transforms

class VideoTensorDataset(Dataset):
    def __init__(
        self,
        data_path:str,
        dataset_name:str="GenVideo",
        generation_model:str="None",
        verbose:bool=False,
        mode:str="train",
        num_frames: int=8,
        transform: Optional[Callable] = None,
        load_len: int=None,
        input_shape: tuple=(224, 224)):
        super().__init__()
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.generation_model = generation_model
        self.verbose = verbose
        self.mode = mode
        self.num_frames = num_frames
        self.input_shape = input_shape
        self.transform = transforms.Compose([
                            transforms.Resize(input_shape),
                            transforms.ToTensor(),
                            transforms.Normalize(
                            mean=[0.485, 0.456, 0.406],  # Mean of ImageNet
                            std=[0.229, 0.224, 0.225]),    # Std of ImageNet
                            ])
        
        generation_models = get_all_generation_models(self.dataset_name)
        self.label = get_label_from_generation_model(self.generation_model)
        
        # different generation AI models, e.g., sora, zeroscope, ...
        assert self.generation_model in generation_models, f"generation model {self.generation_model} is not supported in {self.dataset_name} dataset"
        
        # data_dir
        self.base_dir = os.path.join(
            self.data_path, 'video_frames', self.label, self.generation_model, self.mode
        )

        # all videos path
        self.video_dirs = sorted(
            [os.path.join(self.base_dir, d) for d in os.listdir(self.base_dir)],
            key=lambda x: os.path.basename(x)  # sort by video number
        )

        # all video frames path
        self.video_frame_paths = []
        for video_dir in self.video_dirs:
            # Check if the directory exists and contains .jpg files
            if os.path.isdir(video_dir):
                frames = sorted(
                    [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith('.jpg')],
                    key=lambda x: int(os.path.splitext(os.path.basename(x))[0].replace('frame', ''))
                )
                # Only add the frames list if it is not empty
                if len(frames) == self.num_frames:
                    self.video_frame_paths.append(frames)
        if load_len is not None:
            self.video_frame_paths = self.video_frame_paths[:load_len]
        logger.success(f"[{self.dataset_name} / {self.mode} / {len(self)} videos / {self.generation_model}]")
            
    def __len__(self) -> int:
        return len(self.video_frame_paths)

    def __getitem__(self, idx: int) -> np.ndarray:
        frame_paths = self.video_frame_paths[idx]
        
        video_data = []
        for frame_path in frame_paths:
            img = Image.open(frame_path).convert('RGB')
            
            if self.transform:
                img = self.transform(img)
            
            # convert (H, W, C) to (C, H, W)
            img_array = np.asarray(img)
            video_data.append(img_array)
        
        # merge frames to video
        return np.stack(video_data, axis=0), np.array([0 if self.label=="real" else 1], dtype=np.float32)


class VideoDataset(Dataset):
    def __init__(self, processor, generation_model: str, data_path: str, 
                 mode: str = "train", load_len: int = None, num_frames: int = 8, dataset_name: str = "GenVideo",):
        super().__init__()
        self.processor = processor
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.generation_model = generation_model
        self.mode = mode
        self.num_frames = num_frames
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
        # all video frames path
        self.video_frame_paths = []
        for video_dir in self.video_dirs:
            # Check if the directory exists and contains .jpg files
            if os.path.isdir(video_dir):
                frames = sorted(
                    [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith('.jpg')],
                    key=lambda x: int(os.path.splitext(os.path.basename(x))[0].replace('frame', ''))
                )
                # Only add the frames list if it is not empty
                if len(frames) == self.num_frames:
                    self.video_frame_paths.append(frames)

        if load_len is not None:
            self.video_frame_paths = self.video_frame_paths[:load_len]
        logger.success(f"[{self.dataset_name} / {self.mode} / {len(self)} videos / {self.generation_model}]")
    
    def __len__(self) -> int:
        return len(self.video_frame_paths)
    
    def __getitem__(self, idx):
        frame_paths = self.video_frame_paths[idx]
        
        video_data = []
        for frame_path in frame_paths:
            img = Image.open(frame_path).convert('RGB')
            video_data.append(np.array(img))
        
        video = self.processor(images=video_data, return_tensors="pt").pixel_values[0]
        label = np.array([0 if self.label=="real" else 1], dtype=np.float32)
        # logger.warning(f'{frame_path}, {self.label}, {label}')
        return video, label


def get_video_dataset(data_cfg, mode, processor, load_len=None, generation_model=None, real_model=None, pn_ratio=1):
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
            mode=mode, 
            num_frames=8,
            load_len=load_len,
            )
        real_len = int(load_len / pn_ratio) if load_len else None
        real_dataset = VideoDataset(
            processor=processor,
            data_path=data_cfg.data_path, 
            dataset_name=data_cfg.dataset_name,
            generation_model=real_model,
            mode=mode, 
            num_frames=8,
            load_len=real_len,
            )
    else:
        raise NotImplementedError(f"Feature type {feature_type} is not supported.")
    return ConcatDataset([fake_dataset, real_dataset])


def get_composite_video_dataset(data_cfg, mode, processor, generation_models:list=[], real_model=None):
    feature_type = data_cfg.feature_type
    logger.info(f"Using feature type : {feature_type.upper()}")
    if feature_type == "video":
        fake_datasets = []
        for gen_model in generation_models:
            fake_datasets.append(
                VideoDataset(
                    processor=processor,
                    data_path=data_cfg.data_path, 
                    dataset_name=data_cfg.dataset_name,
                    generation_model=gen_model,
                    mode=mode, 
                    num_frames=8,
                    )
            )
        fake_dataset = ConcatDataset(fake_datasets)
        load_len = len(fake_dataset)
        real_dataset = VideoDataset(
            processor=processor,
            data_path=data_cfg.data_path, 
            dataset_name=data_cfg.dataset_name,
            generation_model=real_model,
            mode=mode, 
            num_frames=8,
            load_len=load_len,
            )
    return ConcatDataset([fake_dataset, real_dataset])


if __name__ == "__main__":
    from omegaconf import OmegaConf
    from transformers import AutoImageProcessor

    print("Testing VideoTensorDataset...")
    dataset = VideoTensorDataset(data_path="/home/ziyuanfang/Data/GenVideo", dataset_name="GenVideo",  generation_model="Sora", mode="test", input_shape=(224,224))
    print(dataset[0][0].shape)
    print(dataset[0][1])

    print("Testing VideoDataset...")
    demo_dataset = VideoDataset(processor=AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-ssv2", use_fast=False), data_path="/home/ziyuanfang/Data/GenVideo", dataset_name="GenVideo",  generation_model="Sora", mode="test")
    print(demo_dataset[0][0].shape)
    # breakpoint()
    print(demo_dataset[0][1])

    print("Testing get_video_dataset...")
    pn_ratio = 1
    cfg = OmegaConf.create({
        'data': {
            'batch_size': 24,
            'val_batch_size': 8,
            'feature_type': 'video',
            'dataset_name': "GenVideo",
            'num_workers': 16,
            'data_path': "/home/ziyuanfang/Data/GenVideo",
            'num_frames': 8,
            'generation_model': "Pika",
            'input_shape': [224,224],
            'train_real_model': "Kinetics-400",
            'val_real_model': "Kinetics-400",
            'test_real_model': "MSR-VTT",
            'train_load_len': 10000,
            'val_load_len': 100,
            'diffuse_steps': 5,
        }
    })
    real_fake_dataset = get_video_dataset(cfg.data, generation_model=cfg.data.generation_model, 
                                          real_model=cfg.data.train_real_model, mode="train", 
                                          load_len=cfg.data.train_load_len, pn_ratio=pn_ratio,
                                          processor=AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-ssv2", use_fast=False))
    print(f"Total training samples: {len(real_fake_dataset)}")
    print(real_fake_dataset[0][0].shape)
    print(real_fake_dataset[0][1])
    