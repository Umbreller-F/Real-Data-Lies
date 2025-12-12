import os
import random
import shutil
from pathlib import Path

def collect_mp4_files(root_dir):
    """Collect all mp4 files from subdirectories."""
    mp4_files = []
    for subdir in sorted(Path(root_dir).iterdir()):
        if subdir.is_dir():
            for file in subdir.glob("*.mp4"):
                mp4_files.append(str(file))
    return sorted(mp4_files)

def split_dataset(file_list, train_size=10000, val_size=1000, test_size=1000, seed=1958):
    """Split files into train/val/test sets with fixed random seed."""
    random.seed(seed)
    shuffled_files = file_list.copy()
    random.shuffle(shuffled_files)
    
    train_files = shuffled_files[:train_size]
    val_files = shuffled_files[train_size:train_size + val_size]
    test_files = shuffled_files[train_size + val_size:train_size + val_size + test_size]
    
    return train_files, val_files, test_files

def save_file_list(file_list, output_path):
    """Save list of files to text file."""
    with open(output_path, 'w') as f:
        for file_path in file_list:
            f.write(f"{Path(file_path).name}\n")

def copy_files_to_target(file_list, target_dir):
    """Copy selected files to target directory."""
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    
    copied_count = 0
    for src_path in file_list:
        src = Path(src_path)
        dst = target_dir / src.name
        shutil.copy2(src, dst)
        copied_count += 1
    
    return copied_count

def main(source_dir, output_dir, copy_target_dir):
    """Main function to process and split the dataset."""
    
    TRAIN_SIZE = 10000
    VAL_SIZE = 1000
    TEST_SIZE = 1000
    SEED = 1958
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(copy_target_dir, exist_ok=True)
    
    # Collect all mp4 files
    print("Collecting mp4 files...")
    all_files = collect_mp4_files(source_dir)
    print(f"Found {len(all_files)} mp4 files")
    
    # Check if we have enough files
    required_total = TRAIN_SIZE + VAL_SIZE + TEST_SIZE
    if len(all_files) < required_total:
        raise ValueError(f"Not enough files. Found {len(all_files)}, need {required_total}")
    
    # Split dataset
    print("Splitting dataset...")
    train_files, val_files, test_files = split_dataset(
        all_files, TRAIN_SIZE, VAL_SIZE, TEST_SIZE, SEED
    )
    
    # Combine all selected files
    all_selected_files = train_files + val_files + test_files
    
    # Save file lists
    print("Saving file lists...")
    save_file_list(train_files, Path(output_dir) / "train_ids.txt")
    save_file_list(val_files, Path(output_dir) / "val_ids.txt")
    save_file_list(test_files, Path(output_dir) / "test_ids.txt")
    
    # Copy all selected files to target directory
    print("Copying all selected files to target directory...")
    total_copied = copy_files_to_target(all_selected_files, copy_target_dir)
    
    print(f"Done! Split {len(all_files)} files into:")
    print(f"  Train: {len(train_files)} files")
    print(f"  Val: {len(val_files)} files")
    print(f"  Test: {len(test_files)} files")
    print(f"  Total selected: {len(all_selected_files)} files")
    print(f"File lists saved to: {output_dir}")
    print(f"All selected files ({total_copied} files) copied to: {copy_target_dir}")

if __name__ == "__main__":
    # Configuration
    SOURCE_DIR = "/data/dataset/genvideo/video/Youku_1M_10s"  # Change this to your source directory
    OUTPUT_DIR = "../Data/GenVideo/split/real/Youku"  # Change this to your output directory
    COPY_TARGET_DIR = "../Data/GenVideo/video/real/Youku"  # Change this to your copy target directory
    
    main(SOURCE_DIR, OUTPUT_DIR, COPY_TARGET_DIR)