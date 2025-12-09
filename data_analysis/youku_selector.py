import os
import random
import shutil

def select_and_copy_videos():
    """Simple version: Select and copy random MP4 files."""
    # Configuration
    source_dir = "/data/dataset/genvideo/video/Youku_1M_10s/0000000_0009999"
    target_dir = "../Data/VQA_videos/Youku"
    txt_dir = "../Data/VQA_videos"
    num_files = 100
    seed = 42
    
    # Set random seed
    random.seed(seed)
    
    # Get all MP4 files
    all_files = [f for f in os.listdir(source_dir) if f.endswith('.mp4')]
    
    if not all_files:
        print("No MP4 files found!")
        return
    
    # Select random files
    selected = random.sample(all_files, min(num_files, len(all_files)))
    
    # Create target directory
    os.makedirs(target_dir, exist_ok=True)
    
    # Copy files and record names
    with open(os.path.join(txt_dir, "Youku_selected.txt"), 'w') as f:
        for filename in selected:
            src = os.path.join(source_dir, filename)
            dst = os.path.join(target_dir, filename)
            shutil.copy2(src, dst)
            f.write(filename + '\n')
    
    print(f"Copied {len(selected)} files to {target_dir}")

if __name__ == "__main__":
    select_and_copy_videos()