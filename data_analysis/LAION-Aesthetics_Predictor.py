import os
import torch
import torch.nn as nn
import clip
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import pytorch_lightning as pl
from glob import glob

# ================= 配置区域 =================

# 数据集根目录 (包含各个模型的文件夹)
ROOT_DIR = "../Data/GenVideo/video_frames" 

# 权重文件路径 (请修改为你本地的实际路径)
WEIGHT_PATH = "data_analysis/improved-aesthetic-predictor/sac+logos+ava1-l14-linearMSE.pth"

# 抽帧间隔 (每隔多少帧计算一次，设为 1 则计算所有帧，设为 5 则每5帧算一次，推荐 5-10 以节省时间)
FRAME_INTERVAL = 4

# 设备配置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===========================================

# 1. 保持原有的 MLP 定义不变
class MLP(pl.LightningModule):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            #nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            #nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

# 2. 保持原有的归一化函数不变 (Numpy 版本)
def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)

def get_image_files(video_dir):
    extensions = ['*.jpg', '*.png', '*.jpeg', '*.bmp']
    files = []
    for ext in extensions:
        files.extend(glob(os.path.join(video_dir, ext)))
    return sorted(files)

def main():
    print(f"Using device: {DEVICE}")
    print(f"Loading weights from: {WEIGHT_PATH}")

    # --- 模型加载 ---
    # 1. 加载 MLP 评分头
    model_mlp = MLP(768) 
    try:
        s = torch.load(WEIGHT_PATH, map_location=DEVICE)
        model_mlp.load_state_dict(s)
    except FileNotFoundError:
        print(f"错误: 找不到权重文件 {WEIGHT_PATH}。请检查路径。")
        return
    
    model_mlp.to(DEVICE)
    model_mlp.eval()

    # 2. 加载 CLIP 模型 (ViT-L/14)
    print("Loading CLIP ViT-L/14...")
    model_clip, preprocess = clip.load("ViT-L/14", device=DEVICE)

    # --- 开始遍历目录 ---
    # 获取一级目录（模型名称）
    if not os.path.exists(ROOT_DIR):
        print(f"错误: 数据集根目录不存在: {ROOT_DIR}")
        return
    final_stats = []

    for label in ['fake', 'real']:
        dir = os.path.join(ROOT_DIR, label)
        model_dirs = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]

        print(f"Found models: {model_dirs}")

        for model_name in model_dirs:
            print(f"\nProcessing Model: {model_name}...")
            if model_name == "VAE":
                continue
            elif model_name in ["Kinetics-400", "Pika", "SEINE"]:
                model_path = os.path.join(dir, model_name, 'val')
            else:
                model_path = os.path.join(dir, model_name, 'test')
            
            # 获取二级目录（视频名称）
            video_dirs = [d for d in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, d))][:100]
            
            all_scores = []
            
            # 遍历视频
            for video_name in tqdm(video_dirs, desc=f"Videos in {model_name}"):
                video_path = os.path.join(model_path, video_name)
                frames = get_image_files(video_path)
                
                if not frames:
                    continue
                    
                # 抽样处理
                sampled_frames = frames[:32:FRAME_INTERVAL]
                
                for frame_path in sampled_frames:
                    try:
                        # 1. 读取并预处理图片
                        pil_image = Image.open(frame_path)
                        image = preprocess(pil_image).unsqueeze(0).to(DEVICE)

                        # 2. CLIP 编码
                        with torch.no_grad():
                            image_features = model_clip.encode_image(image)

                        # 3. 归一化 (保持你的原始逻辑: Tensor -> CPU numpy -> Normalize -> Tensor GPU)
                        # 虽然这有点低效，但为了保证和训练时的预处理一致，我们不动它
                        im_emb_arr = normalized(image_features.cpu().detach().numpy())
                        img_input = torch.from_numpy(im_emb_arr).to(DEVICE).type(torch.float32)

                        # 4. MLP 预测
                        with torch.no_grad():
                            prediction = model_mlp(img_input)
                        
                        score = prediction.item()
                        all_scores.append(score)

                    except Exception as e:
                        print(f"Error processing {frame_path}: {e}")

            # --- 统计该模型的结果 ---
            if all_scores:
                mean_score = np.mean(all_scores)
                std_score = np.std(all_scores)
                
                stats = {
                    'Model': model_name,
                    'Aesthetic_Mean': mean_score,
                    'Aesthetic_Std': std_score,
                    'Frames_Processed': len(all_scores)
                }
                final_stats.append(stats)
                print(f"  -> {model_name} Results: Mean={mean_score:.4f}, Std={std_score:.4f}")
            else:
                print(f"  -> {model_name} has no valid images.")

    # --- 输出最终报告 ---
    print("\n" + "="*50)
    print("FINAL AESTHETIC SCORES REPORT")
    print("="*50)
    
    if final_stats:
        df = pd.DataFrame(final_stats)
        # 按分数高低排序
        df = df.sort_values(by='Aesthetic_Mean', ascending=False)
        print(df.to_string(index=False))
        
        save_path = './data_analysis/aesthetic_scores_genvideo.csv'
        df.to_csv(save_path, index=False)
        print(f"\nReport saved to {save_path}")
    else:
        print("No statistics generated.")

if __name__ == "__main__":
    main()