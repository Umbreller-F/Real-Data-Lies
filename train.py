from utils.experiment_utils import set_seed, seed_worker
from data.dataset import get_paired_dataset
from omegaconf import DictConfig, OmegaConf
from utils.train_utils import *
from models.timesformer import TimeSformer
from models.videomaev2 import VideoMAEv2
from models.demamba import XCLIP_DeMamba, CLIP_DeMamba
from models.dino import DINOv2, DINOv3
from models.npr import resnet50
from models.safe import resnet50_SAFE
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
    
    # region Model
    if 'TimeSformer' in cfg.model.name:
        model = TimeSformer(model_name=cfg.model.name, pretrained=cfg.model.pretrained, freeze_backbone=False)
    elif cfg.model.name == "VideoMAEv2":
        model = VideoMAEv2()
    elif cfg.model.name == "DeMamba":
        if cfg.model.clip:
            model = CLIP_DeMamba()
        else:
            model = XCLIP_DeMamba()
    elif cfg.model.name == "DINOv2":
        model = DINOv2()
        torch.use_deterministic_algorithms(True, warn_only=True)
    elif cfg.model.name == "DINOv3":
        model = DINOv3()
    elif cfg.model.name == "DINOv3-ConvNeXt":
        model = DINOv3('dinov3-convnext-large')
    elif cfg.model.name == "NPR":
        model = resnet50()
    elif cfg.model.name == "SAFE":
        model = resnet50_SAFE()
    else:
        raise NotImplementedError("Model Not supported")
    model = model.to(device)
    if torch.cuda.device_count() >= cfg.trainer.num_gpus and cfg.trainer.num_gpus > 1:
        logger.info(f"Using {cfg.trainer.num_gpus} GPUs for data parallelism.")
        model = nn.DataParallel(model, device_ids=cfg.trainer.device_ids[:cfg.trainer.num_gpus])
    logger.info(f"Model's processor:\n{model.processor}")
    summary(model)
    # endregion

    # region Data
    train_real_model = cfg.data.train_real_model
    train_fake_model = cfg.data.train_fake_model
    val_real_model = cfg.data.val_real_model
    val_fake_model = cfg.data.val_fake_model
    logger.info(f"Loading training data of {train_real_model}-{train_fake_model} and validation data of {val_real_model}-{val_fake_model}")
    pn_ratio = 1
    # train data
    train_dataset = get_paired_dataset(
         cfg.data, processor=model.processor, generation_model=train_fake_model, real_model=train_real_model,
         mode="train", load_len=cfg.data.train_load_len, pn_ratio=pn_ratio,
         num_frames=cfg.data.num_frames, sample_strategy=cfg.data.sample_strategy,
    )

    if cfg.data.data_expansion: # add both LQ and UHQ
        dataset_LQ = get_paired_dataset(cfg.data, processor=model.processor, generation_model='OpenSora', real_model='Kinetics-400',
                                mode="train", load_len=10000, pn_ratio=pn_ratio,
                                num_frames=cfg.data.num_frames, sample_strategy=cfg.data.sample_strategy)
        dataset_UHQ = get_paired_dataset(cfg.data, processor=model.processor, generation_model='DynamicCrafter', real_model='OpenVidHD', 
                                mode="train", load_len=10000, pn_ratio=pn_ratio,
                                num_frames=cfg.data.num_frames, sample_strategy=cfg.data.sample_strategy)
        train_dataset = ConcatDataset([train_dataset, dataset_LQ, dataset_UHQ])
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.data.batch_size, shuffle=True, num_workers=cfg.data.num_workers,
                                worker_init_fn=seed_worker, generator=generator)
    # val data
    val_dataloaders = {}
    val_dataset = get_paired_dataset(cfg.data, "val", generation_model=val_fake_model, real_model=val_real_model, 
                              processor=model.processor, pn_ratio=pn_ratio, load_len=cfg.data.val_load_len,
                              num_frames=cfg.data.num_frames, sample_strategy=cfg.data.sample_strategy)
    val_loader = DataLoader(val_dataset, batch_size=cfg.data.batch_size, shuffle=False, num_workers=cfg.data.num_workers)
    val_dataloaders[f"{val_fake_model}/{val_real_model}"] = val_loader
    # endregion

    # region Train
    global_step = 0
    best_val_auroc = - float("inf")
    best_val_f1 = - float("inf")
    early_stop_patience = cfg.trainer.early_stop_patience
    no_improvement_count = 0
    min_delta = 0.001
    # loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    if cfg.trainer.optimizer.name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=cfg.trainer.optimizer.lr, weight_decay=cfg.trainer.optimizer.weight_decay)
    elif cfg.trainer.optimizer.name == "adamW":
        optimizer = optim.AdamW(model.parameters(), lr=cfg.trainer.optimizer.lr, weight_decay=cfg.trainer.optimizer.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.trainer.optimizer.name}")

    # train logics
    with tqdm(range(cfg.trainer.max_epochs), desc="Epochs", unit="epoch", position=0) as epoch_pbar:
        for epoch in epoch_pbar:
            train_results = train_classifer(model, train_loader, optimizer, device, writer, global_step, criterion, cfg.trainer.max_epochs)
            global_step = train_results["global_step"]
            train_info = " | ".join([f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}"
                         for key, value in train_results.items()])
            
            if (epoch+1) % cfg.trainer.val_check_interval == 0:
                headers, val_results, best_threshold = val_classifer(model, val_dataloaders, criterion, device, writer, global_step)
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