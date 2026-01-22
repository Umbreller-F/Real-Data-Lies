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


if not hasattr(np, 'float'):
    np.float = float

warnings.filterwarnings("ignore", message="Creating a tensor from a list of numpy.ndarrays is extremely slow")
warnings.filterwarnings("ignore", message="`resume_download` is deprecated", category=FutureWarning)


def build_video_augmentor(pipeline):
    """
    Factory function that creates a video augmentation callable.
    
    Args:
        pipeline (ReplayCompose): The albumentations ReplayCompose object.
        
    Returns:
        Callable: A function that takes `video_data` (List[np.ndarray]) as input 
                  and returns augmented frames.
    """
    # Safety Check: Validate the pipeline immediately upon creation
    if not isinstance(pipeline, ReplayCompose):
        raise TypeError("Error: The 'pipeline' argument must be an instance of albumentations.ReplayCompose.")

    def process_frames(video_data):
        """
        The actual function that processes the video frames using the closed-over pipeline.
        """
        augmented_frames = []
        replay_params = None

        for i, frame in enumerate(video_data):
            # Ensure the frame is a Numpy array
            if not isinstance(frame, np.ndarray):
                frame = np.array(frame)

            if i == 0:
                # First frame: Apply augmentation and 'record' the parameters
                data = pipeline(image=frame)
                replay_params = data['replay']
                img_aug = data['image']
            else:
                # Subsequent frames: 'Replay' the exact same parameters
                data = ReplayCompose.replay(replay_params, image=frame)
                img_aug = data['image']
            
            augmented_frames.append(img_aug)

        return augmented_frames

    return process_frames


def build_independent_video_augmentor(pipeline):
    """
    Factory function that creates a video augmentation callable for INDEPENDENT frame processing.
    
    Different random parameters will be applied to each frame (e.g., resulting in flickering 
    for geometric or color augmentations). Useful for noise injection.

    Args:
        pipeline (A.Compose | A.ReplayCompose): The albumentations pipeline. 
            Note: Unlike the consistent augmentor, this works with standard A.Compose too.
        
    Returns:
        Callable: A function that takes `video_data` (List[np.ndarray]) as input 
                  and returns augmented frames.
    """
    # Safety Check: Ensure it's a valid Albumentations pipeline
    if not isinstance(pipeline, (Compose, ReplayCompose)):
        raise TypeError("Error: The 'pipeline' argument must be an instance of albumentations.Compose or ReplayCompose.")

    def process_frames(video_data):
        """
        Processes each frame independently using the pipeline.
        """
        augmented_frames = []

        for frame in video_data:
            # Ensure the frame is a Numpy array
            if not isinstance(frame, np.ndarray):
                frame = np.array(frame)

            # Apply the pipeline directly to the current frame
            # This triggers a new random generation for every single frame
            data = pipeline(image=frame)
            img_aug = data['image']
            
            augmented_frames.append(img_aug)

        return augmented_frames

    return process_frames


class ImageDataset(Dataset):
    def __init__(self, processor, data_path:str, dataset_name:str="GenVideo", generation_model:str="None", frame_sample_rate: int = 4,
                 mode: str = "train", num_frames: int = 8, load_len: int = None, input_shape: tuple = (224, 224),
                 use_aug=False, aug_type=1):
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
        self.use_aug = use_aug
        self.aug_type = aug_type
        if self.use_aug and self.mode == "train":
            if self.aug_type == 1:
                self.aug = va.Sequential([
                    va.Sometimes(0.5, va.HorizontalFlip()),
                    va.Sometimes(0.5, va.VerticalFlip()),
                    va.Sometimes(0.1, va.InvertColor()),
                    va.Sometimes(0.1, va.RandomRotate(degrees=10)),
                    va.Sometimes(1.0, va.GaussianBlur(sigma=0.1)),
                    va.Sometimes(
                        0.1,
                        va.OneOf([
                            va.Salt(ratio=100),
                            va.Pepper(ratio=100)
                        ])
                    )
                ])
            logger.info(f"Use data augmentations.")
        else:
            logger.info("Disable data augmentations.")

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
        '''# all video frames path
        self.image_paths = []
        for video_dir in self.video_dirs:
            # Check if the directory exists and contains .jpg files
            if os.path.isdir(video_dir):
                frames = sorted(
                    [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(('.jpg', '.png'))],
                    key=lambda x: int(os.path.splitext(os.path.basename(x))[0].replace('frame', ''))
                )
                # Only add the frames list if it is not empty
                if len(frames) == self.num_frames:
                    self.image_paths.extend(frames)
            else:
                logger.warning(f"Directory {video_dir} does not exist.")'''
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

        if self.use_aug and self.mode == "train":
            if self.aug_type == 1:
                img = self.aug([img])[0]
        
        if self.processor is None:
            img = self.transform(img)
        else:
            img = self.processor(images=img, return_tensors="pt").pixel_values[0]
        
        label = np.array([0 if self.label=="real" else 1], dtype=np.float32)
        img_id = '/'.join(image_path.split('/')[-2:])
        return img, label, img_id


class VideoDataset(Dataset):
    def __init__(self, processor, generation_model: str, data_path: str, vae = None, sample_strategy='fixed_interval', frame_sample_rate: int = 4,
                 mode: str = "train", load_len: int = None, num_frames: int = 8, dataset_name: str = "GenVideo", no_resize: bool = False, input_shape = (224, 224),
                 use_aug=False, aug_type=1, quality_grl=False, Q_attr=0):
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
        self.no_resize = no_resize
        if self.processor is not None:
            if self.no_resize:
                self.processor.do_resize = False
                if self.mode == "train":
                    self.processor.do_center_crop = False
                    logger.info("Resize disabled. Using random crop for training mode.")
                else:
                    logger.info("Resize disabled. Using center crop for non-training mode.")
            else:
                logger.info("Resize enabled. Follows backbone's standard preprocessing pipeline.")
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
        self.use_aug = use_aug
        self.aug_type = aug_type
        if self.use_aug and self.mode == "train":
            if self.aug_type == 1:
                self.aug = va.Sequential([
                    va.Sometimes(0.5, va.HorizontalFlip()),
                    va.Sometimes(0.5, va.VerticalFlip()),
                    va.Sometimes(0.1, va.InvertColor()),
                    va.Sometimes(0.1, va.RandomRotate(degrees=10)),
                    va.Sometimes(0.1, va.GaussianBlur(sigma=0.1)),
                    va.Sometimes(
                        0.1,
                        va.OneOf([
                            va.Salt(ratio=100),
                            va.Pepper(ratio=100)
                        ])
                    )
                ])
            elif self.aug_type == 2:
                consistent_aug = [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.Rotate(limit=10, p=0.1),
                    A.InvertImg(p=0.1),
                    A.GaussianBlur(sigma_limit=(0.5, 2.5), p=0.1),
                    A.SaltAndPepper(p=0.1),
                ]
                pipe_consistent = ReplayCompose(consistent_aug)
                pipe_consistent.set_random_seed(42)
                self.aug_consistent = build_video_augmentor(pipe_consistent)
                logger.info(f"AUG TYPE 2:\n{pipe_consistent}")
            elif self.aug_type == 3:
                resize_aug = ReplayCompose([
                    OneOf([
                        A.SmallestMaxSize(max_size=720),
                        A.SmallestMaxSize(max_size=360),
                    ])
                ])
                resize_aug.set_random_seed(42)
                self.aug_resize = build_video_augmentor(resize_aug)
                degradation_aug = [
                    OneOf([
                        A.GaussianBlur(sigma_limit=(0.5, 2.5), p=1.0),
                        A.ColorJitter(p=1.0),
                        A.ImageCompression(quality_range=(40, 95), p=1.0),
                    ]),
                ]
                degradation_aug_independent = [
                    OneOf([
                        A.SaltAndPepper(p=1.0),
                        A.GaussNoise(std_range=(0.01, 0.05), p=1.0),
                    ])
                ]
                pipe_degradation_consistent = ReplayCompose(degradation_aug)
                pipe_degradation_consistent.set_random_seed(42)
                self.aug_degradation_consistent = build_video_augmentor(pipe_degradation_consistent)
                pipe_degradation_independent = Compose(degradation_aug_independent)
                pipe_degradation_independent.set_random_seed(42)
                self.aug_degradation_independent = build_independent_video_augmentor(pipe_degradation_independent)
                logger.info(f"AUG TYPE 3:\n{pipe_degradation_consistent}\n{pipe_degradation_independent}")
                '''self.aug = va.Sequential([
                    va.Sometimes(0.5, va.HorizontalFlip()),
                    va.Sometimes(0.5, va.VerticalFlip()),
                    va.Sometimes(0.3, va.InvertColor()),
                    va.Sometimes(0.1, va.RandomRotate(degrees=10)),
                    va.Sometimes(
                        0.3,
                        va.OneOf(
                            va.GaussianBlur(sigma=0.1),
                            va.GaussianBlur(sigma=0.5),
                            va.GaussianBlur(sigma=2.0),
                        )
                    ),
                    # va.Sometimes(0.1, va.GaussianBlur(sigma=0.1)),
                    va.Sometimes(
                        0.1,
                        va.OneOf([
                            va.Salt(ratio=100),
                            va.Pepper(ratio=100)
                        ])
                    )
                ])'''
            elif self.aug_type == 4:
                resize_aug = ReplayCompose([
                    OneOf([
                        A.SmallestMaxSize(max_size=[720, 480, 360, 240]),
                        A.ShiftScaleRotate(rotate_limit=(-30, 30), p=1.0),
                        A.HorizontalFlip(p=1.0),
                        A.VerticalFlip(p=1.0),
                        A.ToGray(p=0.2),
                        A.InvertImg(p=0.2),
                    ])
                ])
                resize_aug.set_random_seed(42)
                self.aug_resize = build_video_augmentor(resize_aug)
                degradation_aug = [
                    OneOf([
                        A.GaussianBlur(sigma_limit=(0.5, 2.5), p=1.0),
                        A.ColorJitter(p=1.0),
                        A.ImageCompression(quality_range=(40, 95), p=1.0),
                        A.MotionBlur(blur_limit=(3, 7), p=1.0),
                    ]),
                ]
                degradation_aug_independent = [
                    OneOf([
                        A.SaltAndPepper(p=1.0),
                        OneOf([
                            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
                            A.GaussNoise(std_range=(0.01, 0.05), p=1.0),
                        ]),
                        A.CoarseDropout(num_holes_range=(3, 8), hole_height_range=(0.05, 0.1), hole_width_range=(0.05, 0.1), fill="inpaint_ns", p=1.0),
                    ])
                ]
                pipe_degradation_consistent = ReplayCompose(degradation_aug)
                pipe_degradation_consistent.set_random_seed(42)
                self.aug_degradation_consistent = build_video_augmentor(pipe_degradation_consistent)
                pipe_degradation_independent = Compose(degradation_aug_independent)
                pipe_degradation_independent.set_random_seed(42)
                self.aug_degradation_independent = build_independent_video_augmentor(pipe_degradation_independent)
                logger.info(f"AUG TYPE 4:\n{pipe_degradation_consistent}\n{pipe_degradation_independent}")
            elif self.aug_type == 5:
                resize_aug = ReplayCompose([
                    OneOf([
                        A.SmallestMaxSize(max_size=[720, 480, 360, 240]),
                        A.ShiftScaleRotate(rotate_limit=(-30, 30), p=1.0),
                        A.HorizontalFlip(p=1.0),
                        A.VerticalFlip(p=1.0),
                        A.ToGray(p=0.2),
                        A.InvertImg(p=0.2),
                    ])
                ])
                resize_aug.set_random_seed(42)
                self.aug_resize = build_video_augmentor(resize_aug)
                degradation_aug = [
                    OneOf([
                        A.GaussianBlur(sigma_limit=(0.5, 2.5), p=1.0),
                        A.ColorJitter(p=1.0),
                        A.ImageCompression(quality_range=(40, 95), p=1.0),
                        A.MotionBlur(blur_limit=(3, 7), p=1.0),
                    ]),
                ]
                degradation_aug_independent = [
                    OneOf([
                        A.SaltAndPepper(p=1.0),
                        OneOf([
                            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
                            A.GaussNoise(std_range=(0.01, 0.05), p=1.0),
                        ]),
                        A.CoarseDropout(num_holes_range=(3, 8), hole_height_range=(0.05, 0.1), hole_width_range=(0.05, 0.1), fill="inpaint_ns", p=1.0),
                    ])
                ]
                pipe_degradation_consistent = ReplayCompose(degradation_aug)
                pipe_degradation_consistent.set_random_seed(42)
                self.aug_degradation_consistent = build_video_augmentor(pipe_degradation_consistent)
                pipe_degradation_independent = Compose(degradation_aug_independent)
                pipe_degradation_independent.set_random_seed(42)
                self.aug_degradation_independent = build_independent_video_augmentor(pipe_degradation_independent)
                logger.info(f"AUG TYPE 4:\n{pipe_degradation_consistent}\n{pipe_degradation_independent}")
            logger.info(f"Use data augmentations.")
        else:
            logger.info("Disable data augmentations.")

        self.input_shape = input_shape
        self.vae = vae
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.generation_model = generation_model
        if self.vae is not None:
            use_vae = True
            self.label = "fake"
        else:
            use_vae = False
            self.label = get_label_from_generation_model(self.generation_model)
        # data_dir
        frames_dir = 'video_frames'
        self.frame_sample_rate = frame_sample_rate
        if not use_vae:
            self.base_dir = os.path.join(
                self.data_path, frames_dir, self.label, self.generation_model, self.mode
            )
        else:
            self.base_dir = os.path.join(
                self.data_path, frames_dir, self.label, 'VAE', self.vae, self.mode
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
        
        self.quality_grl = quality_grl
        if self.quality_grl and self.mode == "train":
            df = pd.read_csv('../Data/RealDist/split/score_gt.csv')
            df.columns = df.columns.str.replace(' ', '')
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str).str.replace(' ', '')
            self.qualtiy_score_groundtruth = {}
            for _, row in df.iterrows():
                self.qualtiy_score_groundtruth[row['video_name']] = row['final_score'] / 100.0
        
        self.Q_attr = Q_attr
        if self.Q_attr in [1, 2]:
            csv_path = f'../Data/RealDist/dover_scores/{self.label}/{self.generation_model}.csv'
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.replace(' ', '')
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str).str.replace(' ', '')
            self.Q_attr_gt = {}
            for _, row in df.iterrows():
                self.Q_attr_gt[row['video_name']] = [row['aesthetic'], row['technical'], row['final_score']]
            if self.mode == 'test' and self.generation_model == 'WildScrape':
                self.Q_attr_gt.pop('D302', None)
                self.video_frame_paths = [paths for paths in self.video_frame_paths if 'D302' not in paths[0]]
        # breakpoint()
        
        if load_len is not None:
            if len(self.video_frame_paths) > load_len:
                logger.info(f"Limiting dataset from {len(self.video_frame_paths)} to {load_len} videos.")
                self.video_frame_paths = self.video_frame_paths[:load_len]
        if not use_vae:
            logger.success(f"[{self.dataset_name} / {self.mode} / {len(self)} videos / {self.generation_model}]")
        else:
            logger.success(f"[{self.dataset_name} / {self.mode} / {len(self)} videos / {self.vae}-VAE]")
    
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
            if self.vae is None:
                video_id = frame_path.split('/')[-2]
            else:
                video_id = f'{self.vae}-VAE#' + frame_path.split('/')[-2]
            return video, label, video_id
        else:
            video_data = []
            for frame_path in frame_paths:
                img = Image.open(frame_path).convert('RGB')
                video_data.append(img)
            
            if self.no_resize:
                width, height = video_data[0].size
                if width < self.input_shape[0] or height < self.input_shape[1]:
                    pad_width = max(0, self.input_shape[0] - width)
                    pad_height = max(0, self.input_shape[1] - height)
                    pad_left = pad_width // 2
                    pad_right = pad_width - pad_left
                    pad_top = pad_height // 2
                    pad_bottom = pad_height - pad_top
                    video_data = [ImageOps.expand(img, border=(pad_left, pad_top, pad_right, pad_bottom), fill=(0, 0, 0)) for img in video_data]
                    width, height = video_data[0].size
                if self.mode == "train":
                    i = random.randint(0, height - self.input_shape[1])
                    j = random.randint(0, width - self.input_shape[0])
                    video_data = [np.array(img.crop((j, i, j + self.input_shape[0], i + self.input_shape[1]))) for img in video_data]
                else:
                    video_data = [np.array(img) for img in video_data]
            else:
                video_data = [np.array(img) for img in video_data]
            
            if self.mode == "train" and self.use_aug:
                if self.aug_type == 1:
                    video_data = self.aug(video_data)
                elif self.aug_type == 2:
                    video_data = self.aug_consistent(video_data)
                elif self.aug_type == 3:
                    # if self.label == "fake":
                    p1 = random.random()
                    if p1 < 0.5:
                        video_data = self.aug_resize(video_data)
                    p2 = random.random()
                    if 0.5 < p2 < 0.8:
                        video_data = self.aug_degradation_consistent(video_data)
                    elif p2 >= 0.8:
                        video_data = self.aug_degradation_independent(video_data)
                elif self.aug_type == 4:
                    p1 = random.random()
                    if p1 < 0.5:
                        video_data = self.aug_resize(video_data)
                    p2 = random.random()
                    if 0.5 < p2 < 0.8:
                        video_data = self.aug_degradation_consistent(video_data)
                    elif p2 >= 0.8:
                        video_data = self.aug_degradation_independent(video_data)
            
            video = self.processor(images=video_data, return_tensors="pt").pixel_values[0]
            label = np.array([0 if self.label=="real" else 1], dtype=np.float32)
            if self.vae is None:
                video_id = frame_path.split('/')[-2]
            else:
                video_id = f'{self.vae}-VAE#' + frame_path.split('/')[-2]
            
            if self.quality_grl and self.mode == "train":
                quality_score = np.array([self.qualtiy_score_groundtruth[video_id]], dtype=np.float32)
                return video, label, video_id, quality_score
            if self.Q_attr in [1, 2]:
                Q_attr_score = np.array(self.Q_attr_gt[video_id], dtype=np.float32)
                return video, label, video_id, Q_attr_score
            return video, label, video_id


class QualityMatchedDataset(VideoDataset):
    def __init__(self, processor, generation_model: str, data_path: str, vae = None, sample_strategy='fixed_interval', frame_sample_rate: int = 4,
                 mode: str = "train", load_len: int = None, num_frames: int = 8, dataset_name: str = "GenVideo", no_resize: bool = False, input_shape = (224, 224)):
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
        self.no_resize = no_resize
        if processor is not None:
            if self.no_resize:
                self.processor.do_resize = False
                if self.mode == "train":
                    self.processor.do_center_crop = False
                    logger.info("Resize disabled. Using random crop for training mode.")
                else:
                    logger.info("Resize disabled. Using center crop for non-training mode.")
            else:
                logger.info("Resize enabled. Follows backbone's standard preprocessing pipeline.")
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
        self.aug = va.Sequential([
            va.Sometimes(0.5, va.HorizontalFlip()),
            va.Sometimes(0.5, va.VerticalFlip()),
            va.Sometimes(0.1, va.InvertColor()),
            va.Sometimes(0.1, va.RandomRotate(degrees=10)),
            va.Sometimes(0.1, va.GaussianBlur(sigma=0.1)),
            va.Sometimes(
                0.1,
                va.OneOf([
                    va.Salt(ratio=100),
                    va.Pepper(ratio=100)
                ])
            )
        ])
        self.input_shape = input_shape
        self.vae = vae
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.generation_model = generation_model
        if self.vae is not None:
            use_vae = True
            self.label = "fake"
        else:
            use_vae = False
            self.label = get_label_from_generation_model(self.generation_model)
        # data_dir
        frames_dir = 'video_frames'
        self.frame_sample_rate = frame_sample_rate
        if not use_vae:
            self.base_dir = os.path.join(
                self.data_path, frames_dir, self.label, self.generation_model, self.mode
            )
        else:
            self.base_dir = os.path.join(
                self.data_path, frames_dir, self.label, 'VAE', self.vae, self.mode
            )
        # all videos path
        # self.video_dirs = sorted(
        #     [os.path.join(self.base_dir, d) for d in os.listdir(self.base_dir)],
        #     key=lambda x: os.path.basename(x)  # sort by video number
        # )
        internvid_path = '../Data/RealDist/video_frames/real/InternVid-AES/train'
        pika_path = '../Data/RealDist/video_frames/fake/Pika/train'
        df = pd.read_csv('../Data/RealDist/split/score_matched.csv')
        df.columns = df.columns.str.replace(' ', '')
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.replace(' ', '')
        self.video_dirs = []
        for _, row in df.iterrows():
            self.video_dirs.append(os.path.join(internvid_path, row['file1']))
            self.video_dirs.append(os.path.join(pika_path, row['file2']))
        
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
            self.video_frame_paths = self.video_frame_paths[:load_len]
        if not use_vae:
            logger.success(f"[{self.dataset_name} / {self.mode} / {len(self)} videos / {self.generation_model}]")
        else:
            logger.success(f"[{self.dataset_name} / {self.mode} / {len(self)} videos / {self.vae}-VAE]")
    
    def __getitem__(self, idx):
        frame_paths = self.video_frame_paths[idx]
        label = "fake" if 'Pika' in frame_paths[0] else "real"
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
            label = np.array([0 if label=="real" else 1], dtype=np.float32)
            if self.vae is None:
                video_id = frame_path.split('/')[-2]
            else:
                video_id = f'{self.vae}-VAE#' + frame_path.split('/')[-2]
            return video, label, video_id
        else:
            video_data = []
            for frame_path in frame_paths:
                img = Image.open(frame_path).convert('RGB')
                video_data.append(img)
            
            if self.no_resize:
                width, height = video_data[0].size
                if width < self.input_shape[0] or height < self.input_shape[1]:
                    pad_width = max(0, self.input_shape[0] - width)
                    pad_height = max(0, self.input_shape[1] - height)
                    pad_left = pad_width // 2
                    pad_right = pad_width - pad_left
                    pad_top = pad_height // 2
                    pad_bottom = pad_height - pad_top
                    video_data = [ImageOps.expand(img, border=(pad_left, pad_top, pad_right, pad_bottom), fill=(0, 0, 0)) for img in video_data]
                    width, height = video_data[0].size
                if self.mode == "train":
                    i = random.randint(0, height - self.input_shape[1])
                    j = random.randint(0, width - self.input_shape[0])
                    video_data = [np.array(img.crop((j, i, j + self.input_shape[0], i + self.input_shape[1]))) for img in video_data]
                else:
                    video_data = [np.array(img) for img in video_data]
            else:
                video_data = [np.array(img) for img in video_data]
            
            if self.mode == "train":
                video_data = self.aug(video_data)
            
            video = self.processor(images=video_data, return_tensors="pt").pixel_values[0]
            label = np.array([0 if label=="real" else 1], dtype=np.float32)
            if self.vae is None:
                video_id = frame_path.split('/')[-2]
            else:
                video_id = f'{self.vae}-VAE#' + frame_path.split('/')[-2]
            return video, label, video_id


def get_paired_dataset(data_cfg, mode, processor, vae=None, recon_prop=0.5, load_len=None, num_frames=8, frame_sample_rate=4, no_resize=False,
                       generation_model=None, real_model=None, pn_ratio=1, sample_strategy='fixed_interval', Q_attr=0,
                       quality_grl=False):
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
    use_aug = data_cfg.get('use_aug', False)
    aug_type = data_cfg.get('aug_type', 1)
    logger.info(f"Using feature type : {feature_type.upper()}")
    if feature_type == "video":
        if vae is not None:
            logger.info(f'Using {vae}-VAE reconstructed samples as fakes, proportion: {recon_prop:.1%}')
            real_dataset = VideoDataset(
                processor=processor,
                data_path=data_cfg.data_path, 
                dataset_name=data_cfg.dataset_name,
                generation_model=real_model,
                sample_strategy=sample_strategy,
                frame_sample_rate=frame_sample_rate,
                no_resize=no_resize,
                mode=mode, 
                num_frames=num_frames,
                load_len=load_len,
                input_shape=tuple(data_cfg.input_shape),
                )
            fake_len = int(len(real_dataset) * pn_ratio)
            vae_len = int(fake_len * recon_prop)
            vae_fake_dataset = VideoDataset(
                processor=processor,
                data_path=data_cfg.data_path, 
                dataset_name=data_cfg.dataset_name,
                generation_model=real_model,
                sample_strategy=sample_strategy,
                frame_sample_rate=frame_sample_rate,
                no_resize=no_resize,
                mode=mode, 
                num_frames=num_frames,
                load_len=vae_len,
                input_shape=tuple(data_cfg.input_shape),
                vae=vae,
                )
            if recon_prop < 1:
                ori_fake_dataset = VideoDataset(
                    processor=processor,
                    data_path=data_cfg.data_path, 
                    dataset_name=data_cfg.dataset_name,
                    generation_model=generation_model,
                    sample_strategy=sample_strategy,
                    frame_sample_rate=frame_sample_rate,
                    no_resize=no_resize,
                    mode=mode, 
                    num_frames=num_frames,
                    load_len=fake_len-vae_len,
                    input_shape=tuple(data_cfg.input_shape),
                    )
                fake_dataset = ConcatDataset([vae_fake_dataset, ori_fake_dataset])
            else:
                fake_dataset = vae_fake_dataset
        else:
            fake_dataset = VideoDataset(
                processor=processor,
                data_path=data_cfg.data_path, 
                dataset_name=data_cfg.dataset_name,
                generation_model=generation_model,
                sample_strategy=sample_strategy,
                frame_sample_rate=frame_sample_rate,
                no_resize=no_resize,
                mode=mode, 
                num_frames=num_frames,
                load_len=load_len,
                input_shape=tuple(data_cfg.input_shape),
                use_aug=use_aug,
                aug_type=aug_type,
                quality_grl=quality_grl,
                Q_attr=Q_attr,
                )
            real_len = int(len(fake_dataset) / pn_ratio)
            real_dataset = VideoDataset(
                processor=processor,
                data_path=data_cfg.data_path, 
                dataset_name=data_cfg.dataset_name,
                generation_model=real_model,
                sample_strategy=sample_strategy,
                frame_sample_rate=frame_sample_rate,
                no_resize=no_resize,
                mode=mode, 
                num_frames=num_frames,
                load_len=real_len,
                input_shape=tuple(data_cfg.input_shape),
                use_aug=use_aug,
                aug_type=aug_type,
                quality_grl=quality_grl,
                Q_attr=Q_attr,
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
            use_aug=use_aug,
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
            use_aug=use_aug,
            )
    else:
        raise NotImplementedError(f"Feature type {feature_type} is not supported.")
    # breakpoint()
    return ConcatDataset([fake_dataset, real_dataset])


def get_single_dataset(data_cfg, mode, processor, load_len=None, num_frames=8, frame_sample_rate=4, no_resize=False,
                       data_model=None, sample_strategy='fixed_interval', Q_attr=0):
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
            no_resize=no_resize,
            mode=mode, 
            num_frames=num_frames,
            load_len=load_len,
            input_shape=tuple(data_cfg.input_shape),
            Q_attr=Q_attr,
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


'''def get_composite_video_dataset(data_cfg, mode, processor, generation_models:list=[], real_model=None, 
                                sample_strategy='fixed_interval',frame_sample_rate=4, no_resize=False,):
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
                    sample_strategy=sample_strategy,
                    frame_sample_rate=frame_sample_rate,
                    no_resize=no_resize,
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
            sample_strategy=sample_strategy,
            frame_sample_rate=frame_sample_rate,
            no_resize=no_resize,
            mode=mode, 
            num_frames=8,
            load_len=load_len,
            )
    return ConcatDataset([fake_dataset, real_dataset])'''


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
            'no_resize': True
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

    # logger.debug("Testing VideoDataset with VAE...")
    # vae_dataset = VideoDataset(processor=AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-ssv2", use_fast=False, local_files_only=True), 
    #                             vae='Wan2.2', data_path="../Data/GenVideo", dataset_name="GenVideo",  generation_model="Sora", mode="train")
    
    # real_fake_dataset = get_video_dataset(cfg.data, generation_model=cfg.data.generation_model, 
    #                                       real_model=cfg.data.train_real_model, mode="train", 
    #                                       load_len=cfg.data.train_load_len, pn_ratio=pn_ratio,
    #                                       vae='Wan2.2', recon_prop=0.5,
    #                                       processor=AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-ssv2", 
    #                                                                                    use_fast=False, local_files_only=True))

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