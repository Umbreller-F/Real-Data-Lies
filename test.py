from utils.experiment_utils import set_seed
from data.dataset_split import GENVIDEO_PIKA, GENVIDEO_SEINE, MYVIDEOS, MYVIDEOS_COMPRESSED, MYVIDEOS_CROPPED, VAE, TEST100, TEST50, REALDIST_PIKA, GENVIDEO_Y_PIKA, REALDIST_I_PIKA, REALDIST_U_PIKA
# from data.video_dataset import get_video_dataset, get_composite_video_dataset
from data.dataset import get_single_dataset
from omegaconf import DictConfig, OmegaConf
from models.timesformer import TimeSformer
from models.videomaev2 import VideoMAEv2
from models.demamba import XCLIP_DeMamba
from models.dino import DINOv2, DINOv3
from models.npr import resnet50
from utils.train_utils import *
from torch.utils.data import DataLoader
from loguru import logger
from tqdm import tqdm
from tabulate import tabulate

import torch.nn as nn
import torch
import pandas as pd
import os
import hydra


@hydra.main(config_path="configs/experiments", config_name="standard-Pika-TALL.yaml", version_base=None)
def test(cfg: DictConfig):
    # region Setup Logging
    log_dir = os.path.join(cfg.log_path, cfg.experiment_name)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{cfg.data.dataset_name}.log")
    logger.add(log_file, format="{time} {level} {message}", level="INFO", rotation="10 MB", compression="zip", mode='w')
    logger.info('Testing configuration:\n' + OmegaConf.to_yaml(cfg))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed)
    # endregion

    # region Load Model
    logger.info(f"Loading model: {cfg.model.name}")
    if 'TimeSformer' in cfg.model.name:
        model = TimeSformer(model_name=cfg.model.name, pretrained=cfg.model.pretrained, freeze_backbone=False)
    elif cfg.model.name == "VideoMAEv2":
        model = VideoMAEv2()
    elif cfg.model.name == "DeMamba":
        model = XCLIP_DeMamba()
    elif cfg.model.name == "DINOv2":
        model = DINOv2()
    elif cfg.model.name == "DINOv3":
        model = DINOv3()
    elif cfg.model.name == "NPR":
        model = resnet50()
    else:
        raise NotImplementedError(f"Model {cfg.model.name} is not supported.")
    # endregion

    # region Load ckpt
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

    # region Load Test Data
    logger.info(f"Loading test data of {cfg.data.dataset_name}...")
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
    # elif cfg.data.dataset_name == "myvideos":
    #     generation_models = MYVIDEOS
    # elif cfg.data.dataset_name == "myvideos_compressed":
    #     generation_models = MYVIDEOS_COMPRESSED
    # elif cfg.data.dataset_name == "myvideos_cropped":
    #     generation_models = MYVIDEOS_CROPPED
    # elif cfg.data.dataset_name == "test100":
    #     generation_models = TEST100
    # elif cfg.data.dataset_name == "test50":
    #     generation_models = TEST50
    
    load_len = cfg.data.test_load_len
    
    test_dataloaders = {}
    data_models = []
    for label in ["real", "fake"]:
        data_models.extend(generation_models[label]["test"])
    
    empty_data_models = []
    for data_model in data_models:
        test_dataset = get_single_dataset(
            cfg.data, mode="test", data_model=data_model, load_len=load_len,
            num_frames=cfg.data.num_frames, sample_strategy=cfg.data.sample_strategy, processor=model.processor, no_resize=cfg.data.no_resize
        )
        if len(test_dataset) == 0:
            logger.warning(f"Test data model {data_model} has no samples, skipping...")
            empty_data_models.append(data_model)
            continue
        test_loader = DataLoader(test_dataset, batch_size=cfg.data.val_batch_size, shuffle=False, num_workers=cfg.data.num_workers)
        test_dataloaders[data_model] = test_loader
    data_models = [dm for dm in data_models if dm not in empty_data_models]
    # endregion

    # region Eval Model
    data_model_results = {}
    results = []

    logger.info("Starting evaluation...")
    for data_model in tqdm(data_models, desc="Testing", unit="data model"):
        data_model_results[data_model] = test_on_dataloader(model, test_dataloaders[data_model], cfg.data.feature_type, device)
    for real_model in generation_models["real"]["test"]:
        if real_model in empty_data_models:
            continue
        for fake_model in generation_models["fake"]["test"]:
            if fake_model in empty_data_models:
                continue
            test_results = calculate_metrics(data_model_results[real_model], data_model_results[fake_model])
            results.append([f"{real_model}-{fake_model}", 
                            test_results["precision"], 
                            test_results["recall"], 
                            test_results["accuracy"], 
                            test_results["f1"],
                            test_results["positive_accuracy"],
                            test_results["negative_accuracy"],
                            test_results["auroc"],
                            ])

    headers = ["Dataset", "Precision", "Recall", "Accuracy", "F1", "FakeACC", "RealACC", "AUROC"]
    # Group results by real_model
    grouped_results = {}
    for row in results:
        dataset_name = row[0]
        if '-' in dataset_name:
            real_model = '-'.join(dataset_name.split('-')[:-1])
            if real_model not in grouped_results:
                grouped_results[real_model] = []
            grouped_results[real_model].append(row)
    # Build final table with averages and separators
    final_results = []
    final_csv = []
    avg_csv = []

    for real_model, model_results in grouped_results.items():
        # Add individual results
        final_results.extend(model_results)
        final_csv.extend(model_results)
        
        # Calculate average for this real_model
        if len(model_results) > 1:
            avg_row = [f"{real_model}-Avg"]
            for i in range(1, len(headers)):
                avg_value = sum(r[i] for r in model_results) / len(model_results)
                avg_row.append(avg_value)
            final_results.append(avg_row)
            final_csv.append(avg_row)
            avg_csv.append(avg_row)
        
        # Add separator (except after last group)
        if real_model != list(grouped_results.keys())[-1]:
            final_csv.append(["-" * 4] * len(headers))
    
    csv_path = os.path.join(log_dir, f"{cfg.save_csv_file}")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    # Save all detailed results to CSV with four decimal places
    df = pd.DataFrame(final_csv, columns=headers)
    df = df.applymap(lambda x: f"{100*x:.2f}" if isinstance(x, float) else x)
    df.to_csv(csv_path, index=False, header=True)
    logger.success(f"Test all detailed results saved to {csv_path}.")
    # Save average values to avg_path
    avg_path = csv_path.replace(".csv", "-avg.csv")
    df_avg = pd.DataFrame(avg_csv, columns=headers)
    df_avg = df_avg.applymap(lambda x: f"{100*x:.2f}" if isinstance(x, float) else x)
    df_avg.to_csv(avg_path, index=False, header=True)
    logger.success(f"Test average results saved to {avg_path}.")

    # Print results in table format
    logger.info("\n" + tabulate(final_results, headers=headers, tablefmt="grid"))
    # endregion

# region Test Func
@torch.no_grad()
def test_on_dataloader(model, test_dataloader, feature_type, device = torch.device('cuda'), frames_per_video = 8):
    model.eval()
    all_labels = []
    all_predicted = []
    all_raw_preds = []

    for batch in tqdm(test_dataloader, desc="Evaluating", leave=False, ncols=100):
        inputs, labels, sample_ids = batch
        inputs, labels = inputs.float().to(device), labels.to(device)

        logits = model(inputs)

        output_pred = logits[:,0].sigmoid().cpu()
        predicted = output_pred > 0.5
        
        # Collect labels and predictions for metric calculation
        all_labels.extend(labels.cpu().numpy().flatten())
        all_predicted.extend(predicted.cpu().numpy().flatten())
        all_raw_preds.extend(output_pred.cpu().numpy().flatten())
    
    all_labels = np.array(all_labels)
    all_predicted = np.array(all_predicted)
    all_raw_preds = np.array(all_raw_preds)
    if feature_type == "video":
        pass
    elif feature_type == "image":
        # Aggregate frame-level predictions to video-level predictions
        num_videos = len(all_labels) // frames_per_video
        all_labels = all_labels.reshape(num_videos, frames_per_video)[:, 0]
        all_raw_preds = all_raw_preds.reshape(num_videos, frames_per_video).mean(axis=1)
        all_predicted = all_raw_preds > 0.5

    return {
        "labels": all_labels,
        "preds": all_predicted,
        "raw_preds": all_raw_preds,
    }

def calculate_metrics(real_model_results, fake_model_results):
    all_labels = np.concatenate([real_model_results['labels'][:len(fake_model_results['labels'])], fake_model_results['labels']])
    all_predicted = np.concatenate([real_model_results['preds'][:len(fake_model_results['preds'])], fake_model_results['preds']])
    all_raw_preds = np.concatenate([real_model_results['raw_preds'][:len(fake_model_results['raw_preds'])], fake_model_results['raw_preds']])
    # Calculate class-wise accuracy
    positive_mask = all_labels == 1
    negative_mask = all_labels == 0
    
    if positive_mask.any():  # Check if there are positive samples
        positive_acc = accuracy_score(all_labels[positive_mask], all_predicted[positive_mask])
    else:
        positive_acc = 0.0
        logger.warning("No positive samples found in test set")
    
    if negative_mask.any():  # Check if there are negative samples
        negative_acc = accuracy_score(all_labels[negative_mask], all_predicted[negative_mask])
    else:
        negative_acc = 0.0
        logger.warning("No negative samples found in test set")

    # Calculate Precision, Recall, and F1 Score using sklearn
    precision = precision_score(all_labels, all_predicted)
    recall = recall_score(all_labels, all_predicted)
    f1 = f1_score(all_labels, all_predicted)
    acc = accuracy_score(all_labels, all_predicted)
    auroc = roc_auc_score(all_labels, all_raw_preds)
        
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": acc,
        "auroc": auroc,
        "positive_accuracy": positive_acc,  # Accuracy on positive class (fake video)
        "negative_accuracy": negative_acc,  # Accuracy on negative class (real video)
    }
# endregion


if __name__ == "__main__":
    test()