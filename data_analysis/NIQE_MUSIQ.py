import os
import torch
import pyiqa
from PIL import Image
from tqdm import tqdm
import pandas as pd
import numpy as np
from glob import glob

# ================= 配置区域 =================
# 数据集根目录路径
# 结构应为: ROOT_DIR / 模型名 / 视频名 / frame0001.png
ROOT_DIR = r'../Data/test100/video_frames' 

# 图片扩展名
EXTS = ('*.jpg', '*.png', '*.jpeg', '*.bmp')

# 是否使用 GPU
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# ================= 模型初始化 =================
print("正在初始化评估模型...")

# 1. 初始化 NIQE (越低越好)
# pyiqa 会自动下载预训练权重
niqe_metric = pyiqa.create_metric('niqe', device=DEVICE)

# 2. 初始化 MUSIQ (美学评分, 越高越好)
# 使用 KonIQ-10k 预训练权重，这是衡量真实感和美学的 SOTA 指标
aesthetic_metric = pyiqa.create_metric('musiq', device=DEVICE)

def get_image_files(video_dir):
    files = []
    for ext in EXTS:
        files.extend(glob(os.path.join(video_dir, ext)))
    return sorted(files)

def evaluate_dataset():
    # 存储最终结果
    model_stats = []
    for label in ['real', 'fake']:
        dir = os.path.join(ROOT_DIR, label)

        # 获取所有一级文件夹（模型名）
        model_names = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        
        for model_name in model_names:
            print(f"\nProcessing Model: {model_name}")
            if model_name == "VAE":
                continue
            elif model_name in ["Kinetics-400", "Pika", "SEINE"]:
                model_path = os.path.join(dir, model_name, 'val')
            else:
                model_path = os.path.join(dir, model_name, 'test')
            # model_path = os.path.join(dir, model_name, 'test')
            
            # 获取该模型下的所有视频文件夹
            video_names = [d for d in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, d))][:100]
            
            niqe_scores_all = []
            aesthetic_scores_all = []
            
            # 遍历视频
            for video_name in tqdm(video_names, desc=f"Videos in {model_name}"):
                video_path = os.path.join(model_path, video_name)
                frames = get_image_files(video_path)
                
                if not frames:
                    continue
                    
                # 为了速度，我们可以选择每个视频抽样计算 (例如每隔5帧算一次)
                # 如果需要全量计算，请把 step 设为 1
                step = 4 
                sampled_frames = frames[:32:step]
                
                for frame_path in sampled_frames:
                    try:
                        # 读取图片 (pyiqa 需要 tensor 输入，0-1之间)
                        # 也可以直接传路径给 pyiqa，但为了控制流，手动读取

                        score_n = niqe_metric(frame_path).item()
                        niqe_scores_all.append(score_n)
                        
                        # 2. 计算 Aesthetics (MUSIQ)
                        score_a = aesthetic_metric(frame_path).item()
                        aesthetic_scores_all.append(score_a)
                            
                    except Exception as e:
                        print(f"Error processing {frame_path}: {e}")

            # 统计该模型的指标
            if niqe_scores_all:
                stats = {
                    'Model': model_name,
                    'NIQE_Mean': np.mean(niqe_scores_all),
                    'NIQE_Std': np.std(niqe_scores_all),
                    'MUSIQ_Mean': np.mean(aesthetic_scores_all),
                    'MUSIQ_Std': np.std(aesthetic_scores_all),
                    'Frame_Count': len(niqe_scores_all)
                }
                model_stats.append(stats)
                print(f"  -> NIQE: {stats['NIQE_Mean']:.4f} | MUSIQ: {stats['MUSIQ_Mean']:.4f}")

    # ================= 输出结果 =================
    if model_stats:
        df = pd.DataFrame(model_stats)
        
        # 格式化一下
        print("\n" + "="*50)
        print("FINAL BENCHMARK RESULTS")
        print("="*50)
        print(df.to_string(index=False))
        
        # 保存到 CSV
        df.to_csv('./data_analysis/test100_quality_report.csv', index=False)
        print("\nReport saved to benchmark_quality_report.csv")
    else:
        print("No valid data found.")

if __name__ == "__main__":
    evaluate_dataset()