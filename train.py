from utils.experiment_utils import set_seed, seed_worker
from data.dataset_split import GENVIDEO_PIKA, GENVIDEO_SEINE, REALDIST_PIKA, GENVIDEO_Y_PIKA, REALDIST_I_PIKA, REALDIST_U_PIKA, REALDIST_O_PIKA, REALDIST_V_PIKA
# from data.video_dataset import get_video_dataset
from data.dataset import get_paired_dataset, QualityMatchedDataset, get_single_dataset
from omegaconf import DictConfig, OmegaConf
from utils.train_utils import *
from models.timesformer import TimeSformer
from models.videomaev2 import VideoMAEv2, VideoMAEv2_Q1
from models.videomaev2_x import VideoMAEv2_X
from models.demamba import XCLIP_DeMamba, XCLIP_DeMamba_Q1, XCLIP_DeMamba_Q2, CLIP_DeMamba
from models.dino import DINOv2, DINOv3
from models.npr import resnet50
from models.tall import TALL_SWIN
from loguru import logger
from tqdm import tqdm
from tabulate import tabulate
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, ConcatDataset
from torchinfo import summary

import torch.nn as nn
import torch.optim as optim
import pandas as pd
import multiprocessing as mp
import hydra
import torch
import time
import os

@hydra.main(config_path="configs/classifier-224x224", config_name="npr.yaml", version_base=None)
def main(cfg: DictConfig):
    time_info = time.strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(cfg.log_path, cfg.experiment_name, time_info)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{cfg.experiment_name}_{time_info}.log")
    logger.add(log_file, format="{time} {level} {message}", level="INFO", rotation="10 MB", compression="zip")
    logger.info('Training configuration:\n' + OmegaConf.to_yaml(cfg))
    writer = SummaryWriter(log_dir=log_dir)

    set_seed(cfg.seed)
    generator = torch.Generator()
    generator.manual_seed(cfg.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    quality_grl = cfg.get('quality_grl', False)
    Q_attr = cfg.model.get('Q_attr', 0)
    
    # region Model
    if 'TimeSformer' in cfg.model.name:
        model = TimeSformer(model_name=cfg.model.name, pretrained=cfg.model.pretrained, freeze_backbone=False)
    elif cfg.model.name == "VideoMAEv2":
        if Q_attr == 0:
            model = VideoMAEv2()
        elif Q_attr == 1:
            model = VideoMAEv2_Q1()
        # if cfg.model.get('extra', False):
        #     model = VideoMAEv2_X()
        # else:
        #     if cfg.model.get('Large', False):
        #         model = VideoMAEv2(model_type='VideoMAEv2-Large')
        #     else:
        #         model = VideoMAEv2()
    elif cfg.model.name == "DeMamba":
        if cfg.model.clip:
            model = CLIP_DeMamba()
        else:
            if Q_attr == 0:
                model = XCLIP_DeMamba(quality_grl=quality_grl)
            elif Q_attr == 1:
                model = XCLIP_DeMamba_Q1()
            elif Q_attr == 2:
                model = XCLIP_DeMamba_Q2()
    elif cfg.model.name == "DINOv2":
        model = DINOv2()
        torch.use_deterministic_algorithms(True, warn_only=True)
    elif cfg.model.name == "DINOv3":
        model = DINOv3()
    elif cfg.model.name == "DINOv3-ConvNeXt":
        model = DINOv3('dinov3-convnext-large')
    elif cfg.model.name == "NPR":
        model = resnet50()
    else:
        raise NotImplementedError("Model Not supported")
    model = model.to(device)
    if torch.cuda.device_count() >= cfg.trainer.num_gpus and cfg.trainer.num_gpus > 1:
        logger.info(f"Using {cfg.trainer.num_gpus} GPUs for data parallelism.")
        model = nn.DataParallel(model, device_ids=cfg.trainer.device_ids[:cfg.trainer.num_gpus])
    logger.info(f"Model's procesor:\n{model.processor}")
    summary(model)
    # endregion

    # region Data
    logger.info(f"Loading training and validation data of {cfg.data.dataset_name}-{cfg.data.generation_model}...")
    if cfg.data.dataset_name == "GenVideo":
        if cfg.data.generation_model == "Pika":
            generation_models = GENVIDEO_PIKA
        elif cfg.data.generation_model == "SEINE":
            generation_models = GENVIDEO_SEINE
    elif cfg.data.dataset_name == "GenVideo-Youku":
        if cfg.data.generation_model == "Pika":
            generation_models = GENVIDEO_Y_PIKA
    elif cfg.data.dataset_name == "RealDist":
        if cfg.data.generation_model == "Pika":
            generation_models = REALDIST_PIKA
    elif cfg.data.dataset_name == "RealDist-I":
        if cfg.data.generation_model == "Pika":
            generation_models = REALDIST_I_PIKA
    elif cfg.data.dataset_name == "RealDist-U":
        if cfg.data.generation_model == "Pika":
            generation_models = REALDIST_U_PIKA
    elif cfg.data.dataset_name == "RealDist-O":
        if cfg.data.generation_model == "Pika":
            generation_models = REALDIST_O_PIKA
    elif cfg.data.dataset_name == "RealDist-V":
        if cfg.data.generation_model == "Pika":
            generation_models = REALDIST_V_PIKA
    else:
        raise NotImplementedError(f"Dataset {cfg.data.dataset_name} is not supported for training.")
    pn_ratio = 1
    # train data
    real_model = generation_models["real"]["train"][0]
    fake_model = generation_models["fake"]["train"][0]
    if cfg.data.get('pika_uniform_match', False):
        fake_model = 'Pika-U'
    if cfg.data.vae_recon:
        mp.set_start_method('spawn', force=True)
        recon_prop = cfg.data.recon_prop
        vae = cfg.data.vae_model
    else:
        vae, recon_prop = None, None
    if cfg.data.get("quality_match", False):
        logger.info('Enable quality matching for training...')
        train_dataset = QualityMatchedDataset(
            processor=model.processor,
            data_path=cfg.data.data_path, 
            dataset_name=cfg.data.dataset_name,
            generation_model=real_model,
            no_resize=cfg.data.no_resize,
            mode="train", 
            num_frames=cfg.data.num_frames,
            sample_strategy=cfg.data.sample_strategy,
            input_shape=tuple(cfg.data.input_shape),
        )
        train_loader = DataLoader(train_dataset, batch_size=cfg.data.batch_size, shuffle=False, num_workers=cfg.data.num_workers,
                                  worker_init_fn=seed_worker, generator=generator)
    else:
        train_dataset = get_paired_dataset(cfg.data, processor=model.processor, generation_model=fake_model, real_model=real_model, 
                                    mode="train", load_len=cfg.data.train_load_len, pn_ratio=pn_ratio,
                                    num_frames=cfg.data.num_frames, sample_strategy=cfg.data.sample_strategy, no_resize=cfg.data.no_resize,
                                    vae=vae, recon_prop=recon_prop, Q_attr=Q_attr,
                                    quality_grl=quality_grl)
        # breakpoint()
        if cfg.data.get("add_degraded_dataset", False):
            degraded_dataset = get_paired_dataset(cfg.data, processor=model.processor, generation_model='Pika-D', real_model='InternVid-AES-D', 
                                    mode="train", load_len=cfg.data.get("num_degraded", 10000), pn_ratio=pn_ratio,
                                    num_frames=cfg.data.num_frames, sample_strategy=cfg.data.sample_strategy, no_resize=cfg.data.no_resize,
                                    vae=vae, recon_prop=recon_prop, Q_attr=Q_attr,
                                    quality_grl=quality_grl)
            train_dataset = ConcatDataset([train_dataset, degraded_dataset])
        if cfg.data.get("add_UHQ_dataset", False):
            UHQ_dataset = get_paired_dataset(cfg.data, processor=model.processor, generation_model='Pika-E-1K', real_model='OpenVid-UHQ', 
                                    mode="train", load_len=1000, pn_ratio=pn_ratio,
                                    num_frames=cfg.data.num_frames, sample_strategy=cfg.data.sample_strategy, no_resize=cfg.data.no_resize,
                                    vae=vae, recon_prop=recon_prop, Q_attr=Q_attr,
                                    quality_grl=quality_grl)
            train_dataset = ConcatDataset([train_dataset, UHQ_dataset])
        if cfg.data.get("add_UHQ_real", False):
            UHQ_real = get_single_dataset(
                cfg.data, mode="train", data_model='OpenVid-UHQ', load_len=1000,
                num_frames=cfg.data.num_frames, sample_strategy=cfg.data.sample_strategy, processor=model.processor, no_resize=cfg.data.no_resize, 
                Q_attr=Q_attr,
            )
            train_dataset = ConcatDataset([train_dataset, UHQ_real])
        if cfg.data.get("add_enhanced_data", False):
            E_dataset = get_paired_dataset(cfg.data, processor=model.processor, generation_model='Pika-E-1K', real_model='InternVid-AES-E-1K', 
                                    mode="train", load_len=1000, pn_ratio=pn_ratio,
                                    num_frames=cfg.data.num_frames, sample_strategy=cfg.data.sample_strategy, no_resize=cfg.data.no_resize,
                                    vae=vae, recon_prop=recon_prop, Q_attr=Q_attr,
                                    quality_grl=quality_grl)
            train_dataset = ConcatDataset([train_dataset, E_dataset])
        if cfg.data.get("K400_1k_type", 0) == 1:
            dataset_1k = get_paired_dataset(cfg.data, processor=model.processor, generation_model='Pika-DE-1k', real_model='K400-1k', 
                                    mode="train", load_len=1000, pn_ratio=pn_ratio,
                                    num_frames=cfg.data.num_frames, sample_strategy=cfg.data.sample_strategy, no_resize=cfg.data.no_resize,
                                    vae=vae, recon_prop=recon_prop, Q_attr=Q_attr,
                                    quality_grl=quality_grl)
            train_dataset = ConcatDataset([train_dataset, dataset_1k])
        if cfg.data.get("K400_1k_type", 0) == 2:
            dataset_1k = get_paired_dataset(cfg.data, processor=model.processor, generation_model='Pika-Rest-1k', real_model='K400-1k', 
                                    mode="train", load_len=1000, pn_ratio=pn_ratio,
                                    num_frames=cfg.data.num_frames, sample_strategy=cfg.data.sample_strategy, no_resize=cfg.data.no_resize,
                                    vae=vae, recon_prop=recon_prop, Q_attr=Q_attr,
                                    quality_grl=quality_grl)
            train_dataset = ConcatDataset([train_dataset, dataset_1k])
        if cfg.data.get('one_more', 0) == 1:
            dataset_one_more = get_paired_dataset(cfg.data, processor=model.processor, generation_model='Pika-2', real_model='Kinetics-400', 
                                    mode="train", load_len=10000, pn_ratio=pn_ratio,
                                    num_frames=cfg.data.num_frames, sample_strategy=cfg.data.sample_strategy, no_resize=cfg.data.no_resize,
                                    vae=vae, recon_prop=recon_prop, Q_attr=Q_attr,
                                    quality_grl=quality_grl)
            train_dataset = ConcatDataset([train_dataset, dataset_one_more])
        if cfg.data.get('one_more', 0) == 2:
            dataset_one_more = get_paired_dataset(cfg.data, processor=model.processor, generation_model='OpenSora', real_model='Kinetics-400', 
                                    mode="train", load_len=10000, pn_ratio=pn_ratio,
                                    num_frames=cfg.data.num_frames, sample_strategy=cfg.data.sample_strategy, no_resize=cfg.data.no_resize,
                                    vae=vae, recon_prop=recon_prop, Q_attr=Q_attr,
                                    quality_grl=quality_grl)
            train_dataset = ConcatDataset([train_dataset, dataset_one_more])
        if cfg.data.get('one_more', 0) == 3:
            dataset_one_more = get_paired_dataset(cfg.data, processor=model.processor, generation_model='SD', real_model='Kinetics-400', 
                                    mode="train", load_len=10000, pn_ratio=pn_ratio,
                                    num_frames=cfg.data.num_frames, sample_strategy=cfg.data.sample_strategy, no_resize=cfg.data.no_resize,
                                    vae=vae, recon_prop=recon_prop, Q_attr=Q_attr,
                                    quality_grl=quality_grl)
            train_dataset = ConcatDataset([train_dataset, dataset_one_more])
        if cfg.data.get('one_more', 0) == 4:
            dataset_one_more = get_paired_dataset(cfg.data, processor=model.processor, generation_model='OpenSora-M', real_model='Kinetics-400-M', 
                                    mode="train", load_len=10000, pn_ratio=pn_ratio,
                                    num_frames=cfg.data.num_frames, sample_strategy=cfg.data.sample_strategy, no_resize=cfg.data.no_resize,
                                    vae=vae, recon_prop=recon_prop, Q_attr=Q_attr,
                                    quality_grl=quality_grl)
            train_dataset = ConcatDataset([train_dataset, dataset_one_more])
        if cfg.data.get('one_more', 0) == 5:
            dataset_one_more = get_paired_dataset(cfg.data, processor=model.processor, generation_model='DynamicCrafter', real_model='Kinetics-400', 
                                    mode="train", load_len=10000, pn_ratio=pn_ratio,
                                    num_frames=cfg.data.num_frames, sample_strategy=cfg.data.sample_strategy, no_resize=cfg.data.no_resize,
                                    vae=vae, recon_prop=recon_prop, Q_attr=Q_attr,
                                    quality_grl=quality_grl)
            train_dataset = ConcatDataset([train_dataset, dataset_one_more])
        if cfg.data.get('one_more', 0) == 6:
            dataset_one_more = get_paired_dataset(cfg.data, processor=model.processor, generation_model='OpenSora-UM', real_model='Kinetics-400-UM', 
                                    mode="train", load_len=10000, pn_ratio=pn_ratio,
                                    num_frames=cfg.data.num_frames, sample_strategy=cfg.data.sample_strategy, no_resize=cfg.data.no_resize,
                                    vae=vae, recon_prop=recon_prop, Q_attr=Q_attr,
                                    quality_grl=quality_grl)
            train_dataset = ConcatDataset([train_dataset, dataset_one_more])

        if cfg.data.get('one_more', 0) == 71:
            dataset_one_more = get_paired_dataset(cfg.data, processor=model.processor, generation_model='OpenSora-MM1', real_model='Kinetics-400-MM1', 
                                    mode="train", load_len=10000, pn_ratio=pn_ratio,
                                    num_frames=cfg.data.num_frames, sample_strategy=cfg.data.sample_strategy, no_resize=cfg.data.no_resize,
                                    vae=vae, recon_prop=recon_prop, Q_attr=Q_attr,
                                    quality_grl=quality_grl)
            train_dataset = ConcatDataset([train_dataset, dataset_one_more])
        if cfg.data.get('one_more', 0) == 73:
            dataset_one_more = get_paired_dataset(cfg.data, processor=model.processor, generation_model='OpenSora-MM3', real_model='Kinetics-400-MM3', 
                                    mode="train", load_len=10000, pn_ratio=pn_ratio,
                                    num_frames=cfg.data.num_frames, sample_strategy=cfg.data.sample_strategy, no_resize=cfg.data.no_resize,
                                    vae=vae, recon_prop=recon_prop, Q_attr=Q_attr,
                                    quality_grl=quality_grl)
            train_dataset = ConcatDataset([train_dataset, dataset_one_more])
        if cfg.data.get('one_more', 0) == 72:
            dataset_one_more = get_paired_dataset(cfg.data, processor=model.processor, generation_model='OpenSora-MM2', real_model='Kinetics-400-MM2', 
                                    mode="train", load_len=10000, pn_ratio=pn_ratio,
                                    num_frames=cfg.data.num_frames, sample_strategy=cfg.data.sample_strategy, no_resize=cfg.data.no_resize,
                                    vae=vae, recon_prop=recon_prop, Q_attr=Q_attr,
                                    quality_grl=quality_grl)
            train_dataset = ConcatDataset([train_dataset, dataset_one_more])
        if cfg.data.get('one_more', 0) == 75:
            dataset_one_more = get_paired_dataset(cfg.data, processor=model.processor, generation_model='OpenSora-MM5', real_model='Kinetics-400-MM5', 
                                    mode="train", load_len=10000, pn_ratio=pn_ratio,
                                    num_frames=cfg.data.num_frames, sample_strategy=cfg.data.sample_strategy, no_resize=cfg.data.no_resize,
                                    vae=vae, recon_prop=recon_prop, Q_attr=Q_attr,
                                    quality_grl=quality_grl)
            train_dataset = ConcatDataset([train_dataset, dataset_one_more])

        if cfg.data.get('one_more', 0) == 100:
            dataset_one_more = get_paired_dataset(cfg.data, processor=model.processor, generation_model='OpenSora-ori', real_model='Kinetics-400-ori', 
                                    mode="train", load_len=10000, pn_ratio=pn_ratio,
                                    num_frames=cfg.data.num_frames, sample_strategy=cfg.data.sample_strategy, no_resize=cfg.data.no_resize,
                                    vae=vae, recon_prop=recon_prop, Q_attr=Q_attr,
                                    quality_grl=quality_grl)
            train_dataset = ConcatDataset([train_dataset, dataset_one_more])
        if cfg.data.get('one_more', 0) == 101:
            dataset_one_more = get_paired_dataset(cfg.data, processor=model.processor, generation_model='OpenSora-ori1', real_model='Kinetics-400-ori1', 
                                    mode="train", load_len=10000, pn_ratio=pn_ratio,
                                    num_frames=cfg.data.num_frames, sample_strategy=cfg.data.sample_strategy, no_resize=cfg.data.no_resize,
                                    vae=vae, recon_prop=recon_prop, Q_attr=Q_attr,
                                    quality_grl=quality_grl)
            train_dataset = ConcatDataset([train_dataset, dataset_one_more])

        if cfg.data.get('one_more', 0) == 81:
            dataset_one_more = get_paired_dataset(cfg.data, processor=model.processor, generation_model='OpenSora', real_model='OpenVidHD', 
                                    mode="train", load_len=10000, pn_ratio=pn_ratio,
                                    num_frames=cfg.data.num_frames, sample_strategy=cfg.data.sample_strategy, no_resize=cfg.data.no_resize,
                                    vae=vae, recon_prop=recon_prop, Q_attr=Q_attr,
                                    quality_grl=quality_grl)
            train_dataset = ConcatDataset([train_dataset, dataset_one_more])
        if cfg.data.get('one_more', 0) == 82:
            dataset_one_more = get_paired_dataset(cfg.data, processor=model.processor, generation_model='DynamicCrafter', real_model='OpenVidHD', 
                                    mode="train", load_len=10000, pn_ratio=pn_ratio,
                                    num_frames=cfg.data.num_frames, sample_strategy=cfg.data.sample_strategy, no_resize=cfg.data.no_resize,
                                    vae=vae, recon_prop=recon_prop, Q_attr=Q_attr,
                                    quality_grl=quality_grl)
            train_dataset = ConcatDataset([train_dataset, dataset_one_more])
        if cfg.data.get('one_more', 0) == 83:
            dataset_one_more = get_paired_dataset(cfg.data, processor=model.processor, generation_model='OpenSora', real_model='Kinetics-400',
                                    mode="train", load_len=10000, pn_ratio=pn_ratio,
                                    num_frames=cfg.data.num_frames, sample_strategy=cfg.data.sample_strategy, no_resize=cfg.data.no_resize,
                                    vae=vae, recon_prop=recon_prop, Q_attr=Q_attr,
                                    quality_grl=quality_grl)
            dataset_two_more = get_paired_dataset(cfg.data, processor=model.processor, generation_model='DynamicCrafter', real_model='OpenVidHD', 
                                    mode="train", load_len=10000, pn_ratio=pn_ratio,
                                    num_frames=cfg.data.num_frames, sample_strategy=cfg.data.sample_strategy, no_resize=cfg.data.no_resize,
                                    vae=vae, recon_prop=recon_prop, Q_attr=Q_attr,
                                    quality_grl=quality_grl)
            train_dataset = ConcatDataset([train_dataset, dataset_one_more, dataset_two_more])
        
        train_loader = DataLoader(train_dataset, batch_size=cfg.data.batch_size, shuffle=True, num_workers=cfg.data.num_workers,
                                  worker_init_fn=seed_worker, generator=generator)
    # val data
    val_dataloaders = {}
    real_model = generation_models["real"]["val"][0]
    fake_model = generation_models["fake"]["val"][0]
    val_dataset = get_paired_dataset(cfg.data, "val", generation_model=fake_model, real_model=real_model, 
                              processor=model.processor, pn_ratio=pn_ratio, load_len=cfg.data.val_load_len,
                              num_frames=cfg.data.num_frames, sample_strategy=cfg.data.sample_strategy, no_resize=cfg.data.no_resize,
                              vae=vae, recon_prop=recon_prop, Q_attr=Q_attr,)
    val_loader = DataLoader(val_dataset, batch_size=cfg.data.batch_size, shuffle=False, num_workers=cfg.data.num_workers)
    val_dataloaders[f"{fake_model}/{real_model}"] = val_loader
    # endregion

    # region Train
    global_step = 0
    best_val_auroc = - float("inf")
    best_val_f1 = - float("inf")
    early_stop_patience = cfg.trainer.get("early_stop_patience", 5)
    no_improvement_count = 0
    min_delta = 0.001
    # loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    if quality_grl:
        quality_score_criterion = nn.MSELoss()
    else:
        quality_score_criterion = None
    if cfg.trainer.optimizer.name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=cfg.trainer.optimizer.lr, weight_decay=cfg.trainer.optimizer.weight_decay)
    elif cfg.trainer.optimizer.name == "adamW":
        optimizer = optim.AdamW(model.parameters(), lr=cfg.trainer.optimizer.lr, weight_decay=cfg.trainer.optimizer.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.trainer.optimizer.name}")

    # train logics
    with tqdm(range(cfg.trainer.max_epochs), desc="Epochs", unit="epoch", position=0) as epoch_pbar:
        for epoch in epoch_pbar:
            train_results = train_classifer(model, train_loader, optimizer, device, writer, global_step, criterion, cfg.trainer.max_epochs, quality_grl, quality_score_criterion, Q_attr)
            global_step = train_results["global_step"]
            train_info = " | ".join([f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}"
                         for key, value in train_results.items()])
            
            if (epoch+1) % cfg.trainer.val_check_interval == 0:
                headers, val_results, best_threshold = val_classifer(model, val_dataloaders, criterion, device, writer, global_step, quality_grl, Q_attr)
                val_info = tabulate(val_results, headers=headers, tablefmt="grid")

                logger.info(
                    f"Epoch {epoch+1:2}/{cfg.trainer.max_epochs:2}\nTrain Info: {train_info}\nValidation Info:\n{val_info}"
                )
                
                val_auroc = val_results[-1][-1]
                val_f1 = val_results[-1][-3]
                
                # Early stopping logic
                if val_f1 > best_val_f1 + min_delta or val_auroc > best_val_auroc + min_delta:
                    no_improvement_count = 0  # Reset counter
                else:
                    no_improvement_count += 1
                    logger.info(f"No improvement, consecutive count: {no_improvement_count}/{early_stop_patience}")
                
                # save epoch model
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "best_threshold": float(best_threshold),
                    "epoch": epoch + 1
                }
                epoch_model_save_path = os.path.join(cfg.save_ckpt_dir, f"ckpt_{str(epoch+1).zfill(3)}.pth")
                os.makedirs(os.path.dirname(epoch_model_save_path), exist_ok=True)
                torch.save(checkpoint, epoch_model_save_path)
                
                # save best model
                if val_f1 > best_val_f1 + min_delta:
                    logger.info(f"Current F1 ({val_f1:.6f}) > Best F1 ({best_val_f1:.6f})")
                    best_val_f1 = val_f1
                    best_model_save_path = os.path.join(cfg.save_ckpt_dir, f"best_f1_ckpt.pth")
                    os.makedirs(os.path.dirname(best_model_save_path), exist_ok=True)
                    torch.save(checkpoint, best_model_save_path)
                    logger.success(f"Model saved at {best_model_save_path}")
                if val_auroc > best_val_auroc + min_delta:
                    logger.info(f"Current AUROC ({val_auroc:.6f}) > Best AUROC ({best_val_auroc:.6f})")
                    best_val_auroc = val_auroc
                    best_model_save_path = os.path.join(cfg.save_ckpt_dir, f"best_auroc_ckpt.pth")
                    os.makedirs(os.path.dirname(best_model_save_path), exist_ok=True)
                    torch.save(checkpoint, best_model_save_path)
                    logger.success(f"Model saved at {best_model_save_path}")

                # Check if training should stop early
                if no_improvement_count >= early_stop_patience:
                    logger.info(f"Validation metric hasn't improved for {early_stop_patience} consecutive epochs, stop training early.")
                    break
            current_train_loss = float(train_results["train_loss"])
            if current_train_loss == 0:
                logger.info(f"Model has perfectly converged: loss={current_train_loss:.4f}, stop training early.")
                break
                
            epoch_pbar.set_postfix({
                **train_results,
            })
            torch.cuda.empty_cache()

    csv_path = os.path.join(cfg.log_path, f"{cfg.experiment_name}/{cfg.data.dataset_name}/{cfg.model.name}_train_results.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df = pd.DataFrame(val_results, columns=headers)
    df.to_csv(csv_path)
    # save model ckpts
    model_save_path = os.path.join(cfg.save_ckpt_dir, f"final_ckpt.pth")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    logger.success(f"Model saved at {model_save_path}")
    # endregion
    
if __name__ == "__main__":
    main()