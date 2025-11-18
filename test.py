from utils.experiment_utils import set_seed
from data.dataset_split import GENVIDEO_PIKA, GENVIDEO_SEINE, MYVIDEOS, MYVIDEOS_COMPRESSED
from data.video_dataset import get_video_dataset, get_composite_video_dataset
from omegaconf import DictConfig, OmegaConf
from models.timesformer import TimesformerBinaryClassifier
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

    # region Load Test Data
    logger.info(f"Loading test data of {cfg.data.dataset_name}...")
    if cfg.data.dataset_name == "GenVideo":
        if cfg.data.generation_model == "Pika":
            generation_models = GENVIDEO_PIKA
        elif cfg.data.generation_model == "SEINE":
            generation_models = GENVIDEO_SEINE
    elif cfg.data.dataset_name == "myvideos":
        generation_models = MYVIDEOS
    elif cfg.data.dataset_name == "myvideos_compressed":
        generation_models = MYVIDEOS_COMPRESSED
    
    load_len = cfg.data.test_load_len
    
    test_dataloaders = {}
    real_model = generation_models["real"]["test"][0]
    for fake_model in generation_models["fake"]["test"]:
        if fake_model == "Sora":
            test_dataset = get_video_dataset(
                cfg.data, mode="test", generation_model=fake_model, real_model=real_model, 
                processor=model.processor, load_len=56
            )
        else:
            test_dataset = get_video_dataset(
                cfg.data, mode="test", generation_model=fake_model, real_model=real_model, 
                processor=model.processor, load_len=load_len
            )
        test_loader = DataLoader(test_dataset, batch_size=cfg.data.val_batch_size, shuffle=True, num_workers=cfg.data.num_workers)
        test_dataloaders[f"{fake_model}"] = test_loader
    # additionally test on composite dataset
    composite_dataset = get_composite_video_dataset(
        cfg.data, mode="test", generation_models=generation_models["fake"]["test"], real_model=real_model, processor=model.processor
        )
    composite_loader = DataLoader(composite_dataset, batch_size=cfg.data.val_batch_size, shuffle=True, num_workers=cfg.data.num_workers)
    # endregion

    # region Evaluate Model
    results = []

    logger.info("Starting evaluation...")
    for name, test_loader in tqdm(test_dataloaders.items(), desc="Testing", unit="dataset"):
        test_results = test_on_dataloader(
            model, test_loader, device
        )
        results.append([name, 
                        test_results["precision"], 
                        test_results["recall"], 
                        test_results["accuracy"], 
                        test_results["f1"],
                        test_results["positive_accuracy"],
                        test_results["negative_accuracy"],
                        test_results["auroc"],
                        ])
        tqdm.write(
            f"Dataset: {name} | Precision: {test_results['precision']:.4f} | Recall: {test_results['recall']:.4f} | "
            f"Accuracy: {test_results['accuracy']:.4f} | F1: {test_results['f1']:.4f} | "
            f"FakeACC: {test_results['positive_accuracy']:.4f} | RealACC: {test_results['negative_accuracy']:.4f} | AUROC: {test_results['auroc']:.4f}"
        )

    headers = ["Dataset", "Precision", "Recall", "Accuracy", "F1", "FakeACC", "RealACC", "AUROC"]
    # Calculate mean of all metrics
    mean_metrics = ["Avg."]
    for i in range(1, len(headers)):
        mean_metrics.append(sum(result[i] for result in results) / len(results))
    results.append(mean_metrics)

    # additionally evaluate on composite dataset
    logger.info("Evaluating on composite dataset...")
    test_results = test_on_dataloader(
        model, composite_loader, device
    )
    results.append([
        "Composite", 
        test_results["precision"], 
        test_results["recall"], 
        test_results["accuracy"], 
        test_results["f1"],
        test_results["positive_accuracy"],
        test_results["negative_accuracy"],
        test_results["auroc"],
    ])
    tqdm.write(
        f"Dataset: {name} | Precision: {test_results['precision']:.4f} | Recall: {test_results['recall']:.4f} | "
        f"Accuracy: {test_results['accuracy']:.4f} | F1: {test_results['f1']:.4f} | "
        f"FakeACC: {test_results['positive_accuracy']:.4f} | RealACC: {test_results['negative_accuracy']:.4f} | AUROC: {test_results['auroc']:.4f}"
    )
    
    csv_path = os.path.join(cfg.log_path, f"{cfg.save_csv_file}")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    # Save results to CSV with four decimal places
    df = pd.DataFrame(results, columns=headers)
    df = df.applymap(lambda x: f"{100*x:.2f}" if isinstance(x, float) else x)
    df = df.transpose()  # Transpose the table
    df.to_csv(csv_path, index=True, header=False)
    logger.success(f"Test results saved to {csv_path}")

    # Print results in table format
    logger.info("\n" + tabulate(results, headers=headers, tablefmt="grid"))
    # endregion

@torch.no_grad()
def test_on_dataloader(model, test_dataloader, device=torch.device('cuda')):
    model.eval()
    all_labels = []
    all_predicted = []
    all_raw_preds = []

    for batch in tqdm(test_dataloader, desc="Evaluating", leave=False, ncols=100):
        inputs, labels, video_ids = batch
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


if __name__ == "__main__":
    test()