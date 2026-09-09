from utils.experiment_utils import set_seed
from data.utils import get_generation_models
from data.dataset_split import RDL
from omegaconf import DictConfig
from models.deep_mmd import deep_MMD
from models.tall import SingleSwinBlockDiscriminator
from utils.train_utils import *
from utils.data_utils import *
from omegaconf import OmegaConf
from loguru import logger
from tqdm import tqdm
import torch.nn as nn
import hydra
from tabulate import tabulate
import torch
import copy
import os
import pandas as pd

@hydra.main(config_path="configs/nsg-vd-224x224", config_name="test.yaml", version_base=None)
def main(cfg: DictConfig):
    # Setup Logging
    log_dir = cfg.log_path
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{cfg.data.dataset_name}.log")
    logger.add(log_file, format="{time} {level} {message}", level="INFO", rotation="10 MB", compression="zip", mode='w')
    logger.info('Testing configuration:\n' + OmegaConf.to_yaml(cfg))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed)

    # region Model
    logger.info(f"Loading model: {cfg.model.name}")
    if cfg.model.name in ["Velocity_Single_TALL_MMD", "Score_Single_TALL_MMD"]:
        logger.info("Using single layer of TALL_MMD model")
        discriminator = SingleSwinBlockDiscriminator(num_features=cfg.model.feature_dim)
    else:
        raise ValueError(f"Unsupported model: {cfg.model.name}")
    
    model = deep_MMD(discriminator=discriminator, 
                        sigma=cfg.model.sigma, 
                        sigma0=cfg.model.sigma0, 
                        epsilon=cfg.model.epsilon, 
                        img_size=cfg.model.img_size, 
                        is_yy_zero=cfg.model.is_yy_zero,
                        is_smooth=cfg.model.is_smooth)
    model.load_state_dict(torch.load(cfg.ckpt_path, weights_only=True))
    if torch.cuda.device_count() >= cfg.trainer.num_gpus and cfg.trainer.num_gpus > 1:
        logger.info(f"Using {cfg.trainer.num_gpus} GPUs for data parallelism.")
        model.net = nn.DataParallel(model.net, device_ids=cfg.trainer.device_ids[:cfg.trainer.num_gpus])
    model = model.to(device)
    model.eval()
    # endregion
    
    # region Data
    if cfg.data.dataset_name == "RDL":
        generation_models = RDL
    else:
        raise NotImplementedError(f"Dataset {cfg.data.dataset_name} is not supported for testing.")

    ref_dataloader = get_ref_dataloader(cfg.data, 
                                    cfg.data.ref_model,
                                    mode=cfg.data.ref_mode,
                                    resolution_size=cfg.data.resolution_size)
    test_dataloaders = {}
    for fake_model in generation_models["fake"]["test"]:
        for real_model in generation_models["real"]["test"]:
            load_len = cfg.data.test_load_len
            test_datasets = get_score_datasets(cfg.data, 
                                              "test",
                                              load_len=load_len,
                                              real_model=real_model,
                                              generation_model=fake_model, filter=False, resolution_size=cfg.data.resolution_size)
            test_loader = get_data_loaders_for_mmd(cfg.data, test_datasets, batch_size=cfg.data.val_batch_size)
            test_dataloaders[f"{real_model}-{fake_model}"] = test_loader

    feature_ref, ref_data = get_ref_features(model, ref_dataloader, cfg.data.ref_load_len)
    feature_ref = feature_ref.cuda()
    regression_model = None
    # endregion
        
    # region Eval Model
    results = []
    logger.info("Starting evaluation...")
    with torch.no_grad():
        for name, test_loader in tqdm(test_dataloaders.items(), desc="Testing", unit="dataset"):
            test_results = test_dMMD(
                model, test_loader, feature_ref, ref_data,
                regression_model
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
    
    csv_path = os.path.join(cfg.log_path, f"{cfg.save_csv_file}")
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
    # Save cross-dataset metric matrix (rows: fake generators, columns: selected real test sets)
    save_cross_dataset_matrix(results, headers, csv_path, metric="AUROC", name_sep="-")

    # Print results in table format
    logger.info("\n" + tabulate(final_results, headers=headers, tablefmt="grid"))
    '''# Save results to CSV
    headers = ["Dataset", "Recall", "Accuracy", "F1", "AUROC", "Precision"]
    # Calculate mean of all metrics
    mean_metrics = ["Avg."]
    ref_result = [cfg.data.ref_load_len]
    for i in range(1, len(headers)):
        mean_metrics.append(sum(result[i] for result in results) / len(results))
        ref_result.append(sum(result[i] for result in results) / len(results))
    results.append(mean_metrics)
    logger.info("\n" + tabulate(results, headers=headers, tablefmt="grid"))
    
    csv_path = os.path.join(cfg.log_path, cfg.save_csv_file)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    # Save results to CSV with four decimal places
    df = pd.DataFrame(results, columns=headers)
    df = df.applymap(lambda x: f"{100*x:.2f}" if isinstance(x, float) else x)
    df = df.transpose()  # Transpose the table
    df.to_csv(csv_path, index=True, header=False)
    logger.success(f"Test results saved to {csv_path}")'''
    # endregion

# region Test Func
@torch.no_grad()
def test_dMMD(model, test_dataloader, feature_ref, ref_data, regression_model=None):
    model.eval()
    is_smooth = model.is_smooth
    sigma = model.sigma
    sigma0_u = model.sigma0_u
    ep = model.ep
    net = model.net    
    
    fake_dataloader = test_dataloader["fake"]
    real_dataloader = test_dataloader["real"]
    dt_clean = []
    dt_adv = []
    with torch.no_grad():
        for idx, (real_data, fake_data) in tqdm(enumerate(zip(real_dataloader, fake_dataloader)), total=len(real_dataloader), desc="Evaluating", leave=False):
            x_real = real_data[0].float().cuda()
            x_fake = fake_data[0].float().cuda()
            
            _,feature_cln = net(x_real,out_feature=True)
            _,feature_adv = net(x_fake,out_feature=True)
            dt_clean.append(MMD_batch2(torch.cat([feature_ref,feature_cln],dim=0), feature_ref.shape[0], torch.cat([ref_data,x_real],dim=0).view(ref_data.shape[0]+x_real.shape[0],-1), sigma, sigma0_u, ep, is_smooth=is_smooth).cpu())
            
            dt_adv.append(MMD_batch2(torch.cat([feature_ref,feature_adv],dim=0), feature_ref.shape[0], torch.cat([ref_data,x_fake],dim=0).view(ref_data.shape[0]+x_fake.shape[0],-1), sigma, sigma0_u, ep, is_smooth=is_smooth).cpu())

        dt_clean = torch.cat(dt_clean)
        dt_adv = torch.cat(dt_adv)
        raw_predict = torch.cat([dt_clean, dt_adv], dim=0)
        if regression_model is None:
            predict = (raw_predict > 1).int()
        else:
            predict = regression_model.predict(raw_predict.reshape(-1,1).cpu())
        labels = torch.cat([torch.zeros(len(dt_clean)), torch.ones(len(dt_adv))], dim=0)
        try:
            auroc = roc_auc_score(labels.cpu(), raw_predict.cpu())
            precision = precision_score(labels, predict)
            recall = recall_score(labels, predict)
            f1 = f1_score(labels, predict)
            acc = accuracy_score(labels, predict)
            # auroc, info = plot_mi(dt_clean, dt_adv, plot=False)
        except ValueError as e:
            logger.error(f"DeepMMD testing failed due to {e}. Exiting the program.")
        
        # Calculate class-wise accuracy
        positive_mask = labels == 1
        negative_mask = labels == 0
        
        if positive_mask.any():  # Check if there are positive samples
            positive_acc = accuracy_score(labels[positive_mask], predict[positive_mask])
        else:
            positive_acc = 0.0
            logger.warning("No positive samples found in test set")
        
        if negative_mask.any():  # Check if there are negative samples
            negative_acc = accuracy_score(labels[negative_mask], predict[negative_mask])
        else:
            negative_acc = 0.0
            logger.warning("No negative samples found in test set")
        
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
    main()