import torch.cuda as cuda 
from tqdm import tqdm
import torch
from utils.mmd_utils import MMD_batch2, plot_mi
import time
import numpy as np
from utils.mmd_utils import MMDu
from loguru import logger
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, precision_recall_curve
import os
import itertools


def train_classifer(model, train_dataloader, optimizer, device, writer, global_step, criterion, max_epochs):
    model.to(device)
    model.train()

    train_loss_history = []
    running_train_loss = 0.0
    
    for batch in tqdm(train_dataloader, desc="Training Progress", position=1, leave=False, total=len(train_dataloader)):
        inputs, labels, video_ids = batch
        if global_step  % 500 == 0:
            writer.add_histogram("train/inputs_distribution", inputs.cpu(), global_step=global_step)
        inputs, labels = inputs.float().to(device), labels.float().to(device)
        logits = model(inputs)
        loss = criterion(logits, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        writer.add_scalar("train/loss", loss.item(), global_step=global_step)
        running_train_loss += loss.item()
        global_step += 1
        # tqdm.write(f"Step: {global_step} Loss: {loss.item()}")
        
    avg_train_loss = running_train_loss / len(train_dataloader)
    train_loss_history.append(avg_train_loss)

    return {
        "train_loss": f"{avg_train_loss:.4f}",
        "global_step": global_step,
    }


@torch.no_grad()
def val_classifer(model, val_dataloaders, loss_fn, device, writer, global_step):
    """
    return:
        results = ["Fake", "Real", "Recall", "Precision", "F1", "Accuracy", "AUROC"]
    """
    model.eval()
    results = []
    for val_name in val_dataloaders:
        val_dataloader = val_dataloaders[val_name]
        # running_val_loss = 0.0
        all_labels = []
        all_predicted = []
        all_raw_preds = []

        start_time = time.time()

        for batch in tqdm(val_dataloader, desc="Evaluating", leave=False, ncols=100):
            inputs, labels, video_ids = batch
            inputs, labels = inputs.float().to(device), labels.to(device)
            logits = model(inputs)
            output_pred = logits[:,0].sigmoid().cpu()
            
            # Collect labels and predictions for metric calculation
            all_labels.extend(labels.cpu().numpy())
            all_raw_preds.extend(output_pred.cpu().numpy())

        # Calculate Precision, Recall, and F1 Score using sklearn
        p, r, thresholds = precision_recall_curve(all_labels, all_raw_preds)
        f1_scores = 2 * (p[:-1] * r[:-1]) / (p[:-1] + r[:-1] + 1e-10)
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
        all_predicted = (np.array(all_raw_preds) >= best_threshold).astype(int)
        logger.info(f"[{val_name}] Best Threshold: {best_threshold:.4f}")
        
        precision = precision_score(all_labels, all_predicted)
        recall = recall_score(all_labels, all_predicted)
        f1 = f1_score(all_labels, all_predicted)
        acc = accuracy_score(all_labels, all_predicted)
        auroc = roc_auc_score(all_labels, all_raw_preds)
        end_time = time.time()
        validation_time = end_time - start_time
        results.append([val_name.split("/")[0], val_name.split("/")[1], recall, precision, f1, acc, auroc])
    headers = ["Fake", "Real", "Recall", "Precision", "F1", "Accuracy", "AUROC"]
    # results.append(["Mean", "Mean", *[sum([x[i] for x in results])/len(results) for i in range(2, 7)]])
    for result in results:
        fake, real = result[0], result[1]
        for header, value in zip(headers[2:], result[2:]):
            writer.add_scalar(f"val/{fake}_{real}_{header}", value, global_step=global_step)
    return headers, results, best_threshold


def train_dMMD(model, train_dataloaders, optimizer, device, global_step, writer):
    model.train()
    
    for batch_idx, (real_data, fake_data) in enumerate(tqdm(zip(train_dataloaders["real"], train_dataloaders["fake"]), desc="Training Progress", position=1, leave=False, total=len(train_dataloaders["real"]))):
        real_data = real_data[0]
        fake_data = fake_data[0]
        if len(real_data) != len(fake_data):
            break
        real_data = real_data.float().to(device, non_blocking=True)
        fake_data = fake_data.float().to(device, non_blocking=True)

        X = torch.cat([real_data, fake_data],dim=0)

        optimizer.zero_grad()
        # Do this within single GPU
        TEMP, ep, sigma, sigma0_u = model(X, real_data.shape[0], writer, global_step)
        
        # Compute Compute J (STAT_u)
        mmd_value_temp = -1 * (TEMP[0])
        mmd_std_temp = torch.sqrt(TEMP[1] + 10 ** (-8))
        STAT_u = torch.div(mmd_value_temp, mmd_std_temp)

        # Compute gradient
        STAT_u.backward()
        optimizer.step()
        model.set_info(mmd_value_temp.item(), ep, sigma, sigma0_u, STAT_u.item())
        writer.add_scalar("train/TEMP", mmd_value_temp.item(), global_step=global_step)
        writer.add_scalar("train/ep", ep, global_step=global_step)
        writer.add_scalar("train/sigma", sigma, global_step=global_step)
        writer.add_scalar("train/sigma0_u", sigma0_u, global_step=global_step)
        writer.add_scalar("train/STAT_u", STAT_u.item(), global_step=global_step)
        global_step += 1
    return {"TEMP": mmd_value_temp.item(),
            "ep": ep,
            "sigma": sigma,
            "sigma0_u": sigma0_u,
            "STAT_u": STAT_u.item(),
            "global_step": global_step}

def train_dMMD_unbalance(model, train_dataloaders, optimizer, device, global_step, writer):
    model.train()
    real_dataloader = train_dataloaders["real"]
    fake_dataloader = train_dataloaders["fake"]
    real_iter = iter(real_dataloader)
    fake_iter = itertools.cycle(fake_dataloader)
    num_unmatched = 0
    num_matched = 0
    for real_data, _ in tqdm(real_iter, desc="Training Progress", position=1, leave=False):
        fake_data, _ = next(fake_iter)
        if len(fake_data) != len(real_data):
            num_unmatched += 1
            continue
        assert len(fake_data) == len(real_data)
        num_matched += 1
        real_data = real_data.float().to(device, non_blocking=True)
        fake_data = fake_data.float().to(device, non_blocking=True)

        X = torch.cat([real_data, fake_data],dim=0)

        optimizer.zero_grad()
        # Do this within single GPU
        TEMP, ep, sigma, sigma0_u = model(X, real_data.shape[0], writer, global_step)
        
        # Compute Compute J (STAT_u)
        mmd_value_temp = -1 * (TEMP[0])
        mmd_std_temp = torch.sqrt(TEMP[1] + 10 ** (-8))
        STAT_u = torch.div(mmd_value_temp, mmd_std_temp)

        # Compute gradient
        STAT_u.backward()
        optimizer.step()
        model.set_info(mmd_value_temp.item(), ep, sigma, sigma0_u, STAT_u.item())
        writer.add_scalar("train/TEMP", mmd_value_temp.item(), global_step=global_step)
        writer.add_scalar("train/ep", ep, global_step=global_step)
        writer.add_scalar("train/sigma", sigma, global_step=global_step)
        writer.add_scalar("train/sigma0_u", sigma0_u, global_step=global_step)
        writer.add_scalar("train/STAT_u", STAT_u.item(), global_step=global_step)
        global_step += 1

    assert (num_matched + num_unmatched) == len(real_dataloader), f"Unmatched: {num_unmatched}, Matched: {num_matched}, Total: {len(real_dataloader)}"
    return {"TEMP": mmd_value_temp.item(),
            "ep": ep,
            "sigma": sigma,
            "sigma0_u": sigma0_u,
            "STAT_u": STAT_u.item(),
            "global_step": global_step}

@torch.no_grad()
def get_ref_features(model, ref_dataloader, ref_len=150):
    model.eval()
    ref_list = []
    for batch_idx, (inputs) in enumerate(ref_dataloader):
        inputs = inputs[0]
        bs = inputs.shape[0]
        if bs * len(ref_list) > ref_len:
            break
        ref_list.append(inputs.float())

    ref_data = torch.cat(ref_list,dim=0).cuda()[:ref_len]
    _,feature_ref = model.net(ref_data,out_feature=True)
    feature_ref = feature_ref.detach().cpu()
    return feature_ref, ref_data

@torch.no_grad()
def get_ref_features_multi_source(model,
                                  ref_dataloaders: dict,
                                  ref_ratio: list,
                                  ref_len: int = 150):
    model.eval()

    if len(ref_dataloaders) != len(ref_ratio):
        raise ValueError("len(ref_dataloaders) != len(ref_ratio)")
    if abs(sum(ref_ratio) - 1.0) > 1e-4:
        raise ValueError("ref_ratio must sum to 1.0")

    names = list(ref_dataloaders.keys())
    target_nums = [round(r * ref_len) for r in ref_ratio]
    assert sum(target_nums) == ref_len
    ref_list = []
    for name, target_num in zip(names, target_nums):
        subset_ref_list = []
        for batch_idx, (inputs) in enumerate(ref_dataloaders[name]):
            inputs = inputs[0]
            bs = inputs.shape[0]
            if bs * len(subset_ref_list) > target_num:
                break
            subset_ref_list.append(inputs.float())
        ref_list.extend(subset_ref_list[:target_num])
    ref_data = torch.cat(ref_list, dim=0).cuda()[:ref_len]
    _, feature_ref = model.net(ref_data, out_feature=True)
    feature_ref = feature_ref.detach().cpu()
    return feature_ref, ref_data



@torch.no_grad()
def val_dMMD(model, val_dataloaders, ref_len, global_step, writer, ref_dataloader=None, ref_dataloaders=None, ref_ratio=None):
    model.eval()
    is_smooth = model.is_smooth
    sigma = model.sigma
    sigma0_u = model.sigma0_u
    ep = model.ep
    net = model.net    
    results = []
    if ref_dataloaders is not None:
        feature_ref, ref_data = get_ref_features_multi_source(model, ref_dataloaders, ref_ratio, ref_len)
    else:
        feature_ref, ref_data = get_ref_features(model, ref_dataloader, ref_len)
    feature_ref = feature_ref.cuda()
    
    for val_name in tqdm(val_dataloaders, desc="Evaluating Different Models", leave=False):
        fake_dataloader = val_dataloaders[val_name]["fake"]
        real_dataloader = val_dataloaders[val_name]["real"]
        dt_clean = []
        dt_adv = []
        with torch.no_grad():
            for real_data, fake_data in tqdm(zip(real_dataloader, fake_dataloader), total=len(real_dataloader), desc="Evaluating", leave=False):
                x_real = real_data[0].float().cuda()
                x_fake = fake_data[0].float().cuda()
                
                _,feature_cln = net(x_real,out_feature=True)
                _,feature_adv = net(x_fake,out_feature=True)

                dt_clean.append(MMD_batch2(torch.cat([feature_ref,feature_cln],dim=0), feature_ref.shape[0], torch.cat([ref_data,x_real],dim=0).view(ref_data.shape[0]+x_real.shape[0],-1), sigma, sigma0_u, ep, is_smooth=is_smooth).cpu())
                
                dt_adv.append(MMD_batch2(torch.cat([feature_ref,feature_adv],dim=0), feature_ref.shape[0], torch.cat([ref_data,x_fake],dim=0).view(ref_data.shape[0]+x_fake.shape[0],-1), sigma, sigma0_u, ep, is_smooth=is_smooth).cpu())

            dt_clean = torch.cat(dt_clean)
            dt_adv = torch.cat(dt_adv)
            try:
                auroc, info = plot_mi(dt_clean, dt_adv, plot=False)
            except ValueError as e:
                logger.error(f"DeepMMD training failed due to {e}. Exiting the program.")
                exit(0)
            results.append([val_name.split("/")[0], val_name.split("/")[1], auroc])
    results.append(["Mean", "Mean", sum([x[2] for x in results])/len(results)])
    headers = ["Fake", "Real", "AUROC"]
    if writer is not None:
        for result in results:
            fake, real = result[0], result[1]
            for header, value in zip(headers[2:], result[2:]):
                writer.add_scalar(f"val/{fake}_{real}_{header}", value, global_step=global_step)
    return results

# region Cross-dataset result matrix
MATRIX_REAL_MODELS = ["Youku", "MSR-VTT", "Vript", "HD-VG-130M", "RealVSR", "UltraVideo"]
MATRIX_REAL_DISPLAY = {"Youku": "Youku-mPLUG"}

def save_cross_dataset_matrix(results, headers, csv_path, metric="AUROC",
                              real_models=None, name_sep="/"):
    """Save an extra fake-generator x real-dataset metric matrix CSV.

    results: rows like [f"{real}{name_sep}{fake}", <metrics aligned with headers[1:]>].
    Rows of the matrix are fake generation models, columns are the selected real
    test sets; the right-most column and the bottom row hold row/column means.
    """
    import pandas as pd
    if real_models is None:
        real_models = MATRIX_REAL_MODELS
    metric_idx = headers.index(metric)

    matrix = {}
    fake_order = []
    for row in results:
        name = row[0]
        parts = name.split(name_sep, 1) if name_sep == "/" else name.rsplit(name_sep, 1)
        if len(parts) != 2:
            continue
        real_model, fake_model = parts
        if real_model not in real_models or fake_model.endswith("-Avg"):
            continue
        if fake_model not in matrix:
            matrix[fake_model] = {}
            fake_order.append(fake_model)
        matrix[fake_model][real_model] = row[metric_idx]

    present_reals = [rm for rm in real_models if any(rm in matrix[fm] for fm in fake_order)]
    if not fake_order or not present_reals:
        logger.warning(f"No test results matched matrix real models {real_models}, skip saving matrix.")
        return

    table = []
    col_values = {rm: [] for rm in present_reals}
    for fm in fake_order:
        row = [fm]
        vals = []
        for rm in present_reals:
            value = matrix[fm].get(rm)
            row.append(value)
            if value is not None:
                vals.append(value)
                col_values[rm].append(value)
        row.append(sum(vals) / len(vals) if vals else None)
        table.append(row)
    avg_row = ["Avg"]
    for rm in present_reals:
        vals = col_values[rm]
        avg_row.append(sum(vals) / len(vals) if vals else None)
    all_vals = [v for vals in col_values.values() for v in vals]
    avg_row.append(sum(all_vals) / len(all_vals) if all_vals else None)
    table.append(avg_row)

    columns = ["Fake \\ Real"] + [MATRIX_REAL_DISPLAY.get(rm, rm) for rm in present_reals] + ["Avg"]
    df = pd.DataFrame(table, columns=columns)
    df = df.applymap(lambda x: "" if x is None or (isinstance(x, float) and pd.isna(x))
                     else (f"{100*x:.2f}" if isinstance(x, (int, float)) else x))
    matrix_path = csv_path.replace(".csv", "-matrix.csv")
    df.to_csv(matrix_path, index=False, header=True)
    logger.success(f"Cross-dataset {metric} matrix saved to {matrix_path}.")
# endregion
