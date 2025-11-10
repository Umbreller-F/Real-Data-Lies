from utils.experiment_utils import set_seed
from data.utils import get_generation_models, get_revised_generation_models
from data.image_dataset import get_image_dataset, get_composite_dataset
from omegaconf import DictConfig, OmegaConf
from models.dino import DINOv2WithLinearProbe, DINOv3WithLinearProbe
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
    if cfg.model.name == "DINOv2":
        model = DINOv2WithLinearProbe('dinov2_vitb14', freeze_backbone=False, num_layers_to_use=None)
    elif cfg.model.name == "DINOv3":
        model = DINOv3WithLinearProbe('dinov3_vitb16', freeze_backbone=False, num_layers_to_use=None)
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
    logger.info("Loading test data...")
    if not cfg.revise:
        generation_models = get_generation_models(cfg.data.dataset_name)
    else:
        generation_models = get_revised_generation_models(cfg.data.dataset_name)
    test_dataloaders = {}
    # Enforces stricter and more rational data isolation, as mandated by our protocol. 
    # Always set cfg.revise to True.
    if not cfg.revise:
        for fake_model in generation_models["fake"]["test"]:
            for real_model in generation_models["real"]["test"]:
                if fake_model == "Sora":
                    load_len = 56
                    test_dataset = get_image_dataset(
                        cfg.data, mode="test", generation_model=fake_model, real_model=real_model, load_len=load_len, input_shape=tuple(cfg.data.input_shape)
                    )
                else:
                    test_dataset = get_image_dataset(
                        cfg.data, mode="test", generation_model=fake_model, real_model=real_model, load_len=cfg.data.test_load_len, input_shape=tuple(cfg.data.input_shape)
                    )
                test_loader = DataLoader(test_dataset, batch_size=cfg.data.batch_size, shuffle=True, num_workers=cfg.data.num_workers)
                test_dataloaders[f"{fake_model}"] = test_loader
    else:
        real_model = cfg.data.test_real_model
        for fake_model in generation_models["fake"]["test"]:
            if fake_model == "Sora":
                load_len = 56
                test_dataset = get_image_dataset(
                    cfg.data, mode="test", generation_model=fake_model, real_model=real_model, load_len=load_len, input_shape=tuple(cfg.data.input_shape)
                )
            else:
                test_dataset = get_image_dataset(
                    cfg.data, mode="test", generation_model=fake_model, real_model=real_model, load_len=cfg.data.test_load_len, input_shape=tuple(cfg.data.input_shape)
                )
            test_loader = DataLoader(test_dataset, batch_size=cfg.data.batch_size, shuffle=True, num_workers=cfg.data.num_workers)
            test_dataloaders[f"{fake_model}"] = test_loader
        # additionally test on composite dataset
        composite_dataset = get_composite_dataset(
            cfg.data, mode="test", generation_models=generation_models["fake"]["test"], real_model=cfg.data.test_real_model, input_shape=tuple(cfg.data.input_shape)
            )
        composite_loader = DataLoader(composite_dataset, batch_size=cfg.data.batch_size, shuffle=True, num_workers=cfg.data.num_workers)
    # endregion

    # region Evaluate Model
    results = []

    logger.info("Starting evaluation...")
    with torch.no_grad():
        for name, test_loader in tqdm(test_dataloaders.items(), desc="Testing", unit="dataset"):
            test_results = test_classifier(
                model, test_loader, device
            )
            results.append([name, 
                            test_results["precision"], 
                            test_results["recall"], 
                            test_results["accuracy"], 
                            test_results["f1"], 
                            test_results["auroc"],
                            ])
            tqdm.write(
                f"Dataset: {name} | Recall: {test_results['recall']:.4f} | F1: {test_results['f1']:.4f} | "
                f"Accuracy: {test_results['accuracy']:.4f} | Precision: {test_results['precision']:.4f} | "
                f"AUROC: {test_results['auroc']:.4f}"
            )

    # Save results to CSV
    # headers = ["Dataset", "Recall", "F1", "Accuracy", "Precision", "Auroc"]
    headers = ["Dataset", "Precision", "Recall", "Accuracy", "F1", "AUROC"]
    # Calculate mean of all metrics
    mean_metrics = ["Avg."]
    for i in range(1, len(headers)):
        mean_metrics.append(sum(result[i] for result in results) / len(results))
    results.append(mean_metrics)

    if cfg.revise:
        # additionally evaluate on composite dataset
        logger.info("Evaluating on composite dataset...")
        test_results = test_classifier(
            model, composite_loader, device
        )
        results.append(["Composite", 
                        test_results["precision"], 
                        test_results["recall"], 
                        test_results["accuracy"], 
                        test_results["f1"], 
                        test_results["auroc"],
                        ])
        tqdm.write(
            f"Dataset: Composite | Recall: {test_results['recall']:.4f} | F1: {test_results['f1']:.4f} | "
            f"Accuracy: {test_results['accuracy']:.4f} | Precision: {test_results['precision']:.4f} | "
            f"AUROC: {test_results['auroc']:.4f}"
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
def test_classifier(model, test_dataloader, device):
    model.eval()
    all_labels = []
    all_predicted = []
    all_raw_preds = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating", leave=False, ncols=100):
            inputs, labels = batch
            inputs, labels = inputs.float().to(device), labels.to(device)

            logits = model(inputs)

            output_pred = logits[:,0].sigmoid().cpu()
            predicted = output_pred > 0.5
            
            # Collect labels and predictions for metric calculation
            all_labels.extend(labels.cpu().numpy())
            all_predicted.extend(predicted.cpu().numpy())
            all_raw_preds.extend(output_pred.cpu().numpy())

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
    }


if __name__ == "__main__":
    test()