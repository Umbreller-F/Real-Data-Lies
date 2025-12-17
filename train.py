from utils.experiment_utils import set_seed
from data.dataset_split import GENVIDEO_PIKA, GENVIDEO_SEINE, REALDIST_PIKA, GENVIDEO_Y_PIKA
# from data.video_dataset import get_video_dataset
from data.dataset import get_dataset
from omegaconf import DictConfig, OmegaConf
from utils.train_utils import *
from models.timesformer import TimesformerBinaryClassifier
from models.demamba import XCLIP_DeMamba
from models.dino import DINOv2WithLinearProbe, DINOv3WithLinearProbe
from models.npr import resnet50
from loguru import logger
from tqdm import tqdm
from tabulate import tabulate
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
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
    log_dir = os.path.join(cfg.log_path, cfg.experiment_name)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{cfg.model.name}_{time.strftime('%Y%m%d_%H%M%S')}.log")
    logger.add(log_file, format="{time} {level} {message}", level="INFO", rotation="10 MB", compression="zip")
    logger.info('Training configuration:\n' + OmegaConf.to_yaml(cfg))
    writer = SummaryWriter(log_dir=log_dir)
    set_seed(cfg.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # region Model
    if 'timesformer' in cfg.model.name:
        model = TimesformerBinaryClassifier(model_name=cfg.model.name, pretrained=cfg.model.pretrained, freeze_backbone=False)
    elif cfg.model.name == "DeMamba":
        model = XCLIP_DeMamba()
    elif cfg.model.name == "DINOv2":
        model = DINOv2WithLinearProbe('dinov2_vitb14', freeze_backbone=False, num_layers_to_use=None)
    elif cfg.model.name == "DINOv3":
        model = DINOv3WithLinearProbe('dinov3_vitb16', freeze_backbone=False, num_layers_to_use=None)
    elif cfg.model.name == "NPR":
        model = resnet50()
    else:
        raise NotImplementedError("Model Not supported")
    model = model.to(device)
    if torch.cuda.device_count() >= cfg.trainer.num_gpus and cfg.trainer.num_gpus > 1:
        logger.info(f"Using {cfg.trainer.num_gpus} GPUs for data parallelism.")
        model = nn.DataParallel(model, device_ids=cfg.trainer.device_ids[:cfg.trainer.num_gpus])

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
    else:
        raise NotImplementedError(f"Dataset {cfg.data.dataset_name} is not supported for training.")
    pn_ratio = 1
    # train data
    real_model = generation_models["real"]["train"][0]
    fake_model = generation_models["fake"]["train"][0]
    if cfg.data.vae_recon:
        mp.set_start_method('spawn', force=True)
        recon_prop = cfg.data.recon_prop
        vae = cfg.data.vae_model
    else:
        vae, recon_prop = None, None
    train_dataset = get_dataset(cfg.data, processor=model.processor, generation_model=fake_model, real_model=real_model, 
                                mode="train", load_len=cfg.data.train_load_len, pn_ratio=pn_ratio,
                                sample_strategy=cfg.data.sample_strategy, no_resize=cfg.data.no_resize,
                                vae=vae, recon_prop=recon_prop)
    train_loader = DataLoader(train_dataset, batch_size=cfg.data.batch_size, shuffle=True, num_workers=cfg.data.num_workers)
    # val data
    val_dataloaders = {}
    real_model = generation_models["real"]["val"][0]
    fake_model = generation_models["fake"]["val"][0]
    val_dataset = get_dataset(cfg.data, "val", generation_model=fake_model, real_model=real_model, 
                              processor=model.processor, pn_ratio=pn_ratio, load_len=cfg.data.val_load_len,
                              sample_strategy=cfg.data.sample_strategy, no_resize=cfg.data.no_resize,
                              vae=vae, recon_prop=recon_prop)
    val_loader = DataLoader(val_dataset, batch_size=cfg.data.batch_size, shuffle=True, num_workers=cfg.data.num_workers)
    val_dataloaders[f"{fake_model}/{real_model}"] = val_loader
    # endregion

    # region Train
    global_step = 0
    best_val_auroc = - float("inf")
    best_val_acc = - float("inf")
    early_stop_patience = 5
    no_improvement_count = 0
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
            train_results = train_classifer(model, train_loader, optimizer, criterion, device, writer, global_step, val_dataloaders, criterion, cfg)
            global_step = train_results["global_step"]
            train_info = " | ".join([f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}"
                         for key, value in train_results.items()])
            
            if (epoch+1) % cfg.trainer.val_check_interval == 0:
                headers, val_results = val_classifer(model, val_dataloaders, criterion, device, writer, global_step)
                val_info = tabulate(val_results, headers=headers, tablefmt="grid")

                logger.info(
                    f"Epoch {epoch+1:2}/{cfg.trainer.max_epochs:2}\nTrain Info: {train_info}\nValidation Info:\n{val_info}"
                )
                
                val_auroc = val_results[-1][-1]
                val_acc = val_results[-1][-3]
                # Early stopping logic
                if val_acc > best_val_acc or val_auroc > best_val_auroc:
                    no_improvement_count = 0  # Reset counter
                else:
                    no_improvement_count += 1
                    logger.info(f"No improvement, consecutive count: {no_improvement_count}/{early_stop_patience}")
                # save best model
                if val_acc > best_val_acc:
                    logger.info(f"Current acc ({val_acc:.4f}) > Best acc ({best_val_acc:.4f})")
                    best_val_acc = val_acc
                    best_model_save_path = os.path.join(cfg.save_ckpt_dir, f"best_acc_ckpt.pth")
                    os.makedirs(os.path.dirname(best_model_save_path), exist_ok=True)
                    torch.save(model.state_dict(), best_model_save_path)
                    logger.success(f"Model saved at {best_model_save_path}")
                if val_auroc > best_val_auroc:
                    logger.info(f"Current auroc ({val_auroc:.4f}) > Best auroc ({best_val_auroc:.4f})")
                    best_val_auroc = val_auroc
                    best_model_save_path = os.path.join(cfg.save_ckpt_dir, f"best_auroc_ckpt.pth")
                    os.makedirs(os.path.dirname(best_model_save_path), exist_ok=True)
                    torch.save(model.state_dict(), best_model_save_path)
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