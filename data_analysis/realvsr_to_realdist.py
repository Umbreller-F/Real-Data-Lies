import os
import shutil
from pathlib import Path

def copy_mov_files(source_dir, target_dir, list_file, max_files=1000):
    """Copy first 1000 MOV files from source to target directory."""
    
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    list_path = Path(list_file)
    
    # Create target directory if it doesn't exist
    target_path.mkdir(parents=True, exist_ok=True)
    
    # Create list file directory if it doesn't exist
    list_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get all MOV files and sort by name
    mov_files = sorted(source_path.glob("*.mov"))
    
    # Take first 1000 files
    selected_files = mov_files[:max_files]
    
    # Copy files and collect names
    copied_names = []
    for file in selected_files:
        target_file = target_path / file.name
        shutil.copy2(file, target_file)
        copied_names.append(file.name)  # Keep full filename with extension
    
    # Write filenames to text file
    with open(list_file, 'w') as f:
        f.write('\n'.join(copied_names))
    
    print(f"Copied {len(copied_names)} files to {target_dir}")
    print(f"File list saved to {list_file}")

if __name__ == "__main__":
    # Configure your paths here
    SOURCE_DIR = "../Data/RealVSR/videos"
    TARGET_DIR = "../Data/RealDist/video/real/RealVSR"
    LIST_FILE = "../Data/RealDist/split/real/RealVSR/test_ids.txt"
    
    copy_mov_files(SOURCE_DIR, TARGET_DIR, LIST_FILE, max_files=1000)