import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset
from data.utils import *
from pprint import pprint as print
from loguru import logger
from torchvision import transforms

class ImageDataset(Dataset):
    def __init__(
        self,
        data_path:str,
        dataset_name:str="GenVideo",
        generation_model:str="None",
        mode:str="train",
        num_frames: int=8,
        load_len: int=None,
        input_shape: tuple=(224, 224)):
        super().__init__()
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.generation_model = generation_model
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
        """
        """
        
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

        # limit dataset size
        if load_len is not None:
            logger.info(f"Limiting dataset from {len(self.video_dirs)} to {load_len} videos.")
            self.video_dirs = self.video_dirs[:load_len]

        # all video frames path
        self.image_paths = []
        for video_dir in self.video_dirs:
            # Check if the directory exists and contains .jpg files
            if os.path.isdir(video_dir):
                frames = sorted(
                    [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith('.jpg')],
                    key=lambda x: int(os.path.splitext(os.path.basename(x))[0].replace('frame', ''))
                )
                # Only add the frames list if it is not empty
                if len(frames) == self.num_frames:
                    self.image_paths.extend(frames)
            else:
                logger.warning(f"Directory {video_dir} does not exist.")
        logger.success(f"Dataset initialized: {self.dataset_name} / {self.mode} / {len(self.image_paths)} images / {len(self.video_dirs)} videos / {self.generation_model}")
            
    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> np.ndarray:

        image_path = self.image_paths[idx]
        
        img = Image.open(image_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        # convert (H, W, C) to (C, H, W)
        img_array = np.asarray(img)
        
        # merge frames to video
        return img_array, np.array([0 if self.label=="real" else 1], dtype=np.float32)

def get_image_dataset(data_cfg, mode, load_len=1000, generation_model=None, real_model=None, pn_ratio=1, input_shape=(224,224)):
    feature_type = data_cfg.feature_type
    logger.info(f"Using feature type : {feature_type.upper()}")
    if feature_type == "image":
        real_dataset = ImageDataset(
            data_path=data_cfg.data_path, 
            dataset_name=data_cfg.dataset_name,
            generation_model=real_model,
            mode=mode, 
            num_frames=8,
            input_shape=input_shape,
            load_len=load_len,
            )
        fake_len = int(load_len * pn_ratio)
        fake_dataset = ImageDataset(
            data_path=data_cfg.data_path, 
            dataset_name=data_cfg.dataset_name,
            generation_model=generation_model,
            mode=mode, 
            num_frames=8,
            input_shape=input_shape,
            load_len=fake_len,
            )
    return ConcatDataset([fake_dataset, real_dataset])


def get_composite_dataset(data_cfg, mode, generation_models:list=[], real_model=None, input_shape=(224,224)):
    feature_type = data_cfg.feature_type
    logger.info(f"Using feature type : {feature_type.upper()}")
    if feature_type == "image":
        real_dataset = ImageDataset(
            data_path=data_cfg.data_path, 
            dataset_name=data_cfg.dataset_name,
            generation_model=real_model,
            mode=mode, 
            num_frames=8,
            input_shape=input_shape,
            )
        fake_datasets = []
        for gen_model in generation_models:
            fake_datasets.append(
                ImageDataset(
                    data_path=data_cfg.data_path, 
                    dataset_name=data_cfg.dataset_name,
                    generation_model=gen_model,
                    mode=mode, 
                    num_frames=8,
                    input_shape=input_shape,
                    )
            )
        fake_dataset = ConcatDataset(fake_datasets)
    return ConcatDataset([fake_dataset, real_dataset])


if __name__ == "__main__":
    # python -m data.image_dataset
    from omegaconf import OmegaConf
    print("Testing ImageDataset...")
    dataset = ImageDataset(data_path="../Data/GenVideo", dataset_name="GenVideo",  generation_model="Sora", mode="test", input_shape=(224,224))
    print(dataset[0][0].shape)
    print(dataset[0][1])
    print(len(dataset))

    print("Testing get_image_dataset...")
    pn_ratio = 1
    cfg = OmegaConf.create({
        'data': {
            'batch_size': 24,
            'val_batch_size': 8,
            'feature_type': 'image',
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
        }
    })
    real_fake_dataset = get_image_dataset(cfg.data, generation_model=cfg.data.generation_model, 
                                          real_model=cfg.data.train_real_model, mode="train", 
                                          load_len=cfg.data.train_load_len, pn_ratio=pn_ratio)
    print(f"Total training samples: {len(real_fake_dataset)}")
    print(real_fake_dataset[0][0].shape)
    print(real_fake_dataset[0][1])

    print("Testing get_composite_dataset...")
    composite_dataset = get_composite_dataset(cfg.data, mode="test", 
                                              generation_models=[
                                                "ModelScope", 
                                                "MorphStudio",  
                                                "MoonValley", 
                                                "HotShot",
                                                "Show_1",
                                                "Gen2", 
                                                "Crafter",
                                                "Lavie", 
                                                "Sora", 
                                                "WildScrape"
                                              ], 
                                              real_model=cfg.data.test_real_model,
                                              input_shape=(256, 256))
    print(f"Total samples: {len(composite_dataset)}")
    print(composite_dataset[0][0].shape)
    print(composite_dataset[0][1])
    