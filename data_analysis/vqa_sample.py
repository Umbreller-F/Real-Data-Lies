import os
import random
import shutil
from pathlib import Path


YOUKU_ERROR_VIDEOS = [
    'yplug_pre_train_0441389_28_10.mp4',
    'yplug_pre_train_0507575_18_10.mp4',
    'yplug_pre_train_0600057_10_10.mp4',
    'yplug_pre_train_0675177_70_10.mp4'
]

def sample_videos(source_root, target_root, num_samples=100, random_seed=42):
    """
    Randomly sample videos from each subdirectory with fixed random seed
    
    Parameters:
    source_root: Source directory containing 'fake' and 'real' folders
    target_root: Target directory for sampled videos
    num_samples: Number of videos to sample from each subdirectory
    random_seed: Random seed for reproducibility
    """
    # Set fixed random seed for reproducibility
    random.seed(random_seed)
    
    source_path = Path(source_root)
    target_path = Path(target_root)
    
    # Create target directory
    target_path.mkdir(parents=True, exist_ok=True)
    
    # Supported video formats
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v', '.mpg', '.mpeg'}
    
    # Summary log
    summary_log = []
    
    print("Starting video sampling...")
    print("=" * 50)
    print(f"Random seed: {random_seed}")
    print(f"Samples per directory: {num_samples}")
    print("=" * 50)
    
    # Process 'fake' and 'real' directories
    for top_dir in ['fake', 'real']:
        top_dir_path = source_path / top_dir
        if not top_dir_path.exists():
            print(f"Warning: Directory '{top_dir}' not found")
            continue
            
        print(f"\nProcessing '{top_dir}' directory:")
        print("-" * 30)
        
        # Process each subdirectory
        for subdir in sorted(top_dir_path.iterdir()):
            if not subdir.is_dir():
                continue
                
            subdir_name = subdir.name
            print(f"  Processing: {subdir_name}")
            
            # Get all video files
            all_videos = []
            for root, dirs, files in os.walk(subdir):
                for file in files:
                    if Path(file).suffix.lower() in video_extensions:
                        all_videos.append(Path(root) / file)
            
            if not all_videos:
                print(f"    No video files found, skipping")
                summary_log.append(f"{subdir_name}: 0 files (no videos)")
                continue
            
            # Sort for consistent ordering before sampling
            all_videos = sorted(all_videos)
            
            # Random sampling
            total_videos = len(all_videos)
            samples_to_take = min(num_samples, total_videos)
            
            if total_videos <= num_samples:
                selected_videos = all_videos
                print(f"    Found {total_videos} videos, taking all")
            else:
                # Use random.sample for sampling without replacement
                selected_indices = random.sample(range(total_videos), samples_to_take)
                selected_videos = [all_videos[i] for i in sorted(selected_indices)]
                print(f"    Found {total_videos} videos, sampling {samples_to_take}")
            
            # Create target subdirectory
            target_subdir = target_path / subdir_name
            target_subdir.mkdir(exist_ok=True)
            
            # Copy files and record names
            selected_filenames = []
            copied_count = 0
            
            for src_file in selected_videos:
                # Target file path
                dst_file = target_subdir / src_file.name
                if dst_file.name in YOUKU_ERROR_VIDEOS:
                    continue
                # Handle duplicate filenames
                if dst_file.exists():
                    print(f"    Skip {src_file.name} (already exists in target)")
                    continue
                
                # Copy file
                try:
                    shutil.copy2(src_file, dst_file)
                    copied_count += 1
                    
                    # Record relative path
                    rel_path = src_file.relative_to(subdir)
                    selected_filenames.append(str(rel_path))
                    
                except Exception as e:
                    print(f"    Error copying {src_file.name}: {e}")
            
            # Save selected filenames to txt file (simple format)
            if selected_filenames:
                txt_filename = target_path / f"{subdir_name}_selected.txt"
                with open(txt_filename, 'w', encoding='utf-8') as f:
                    # Write one filename per line, no extra information
                    for filename in selected_filenames:
                        f.write(f"{filename}\n")
                
                print(f"    Saved file list to: {txt_filename.name}")
            
            print(f"    Copied {copied_count} files to {target_subdir.name}/")
            
            # Add to summary
            summary_log.append(f"{subdir_name}: {copied_count} files ({samples_to_take}/{total_videos})")
    
    # Save summary file
    if summary_log:
        summary_file = target_path / "sampling_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"Random seed: {random_seed}\n")
            f.write(f"Samples per directory: {num_samples}\n")
            f.write("=" * 40 + "\n")
            
            for entry in summary_log:
                f.write(f"{entry}\n")
        
        print(f"\nSummary saved to: {summary_file}")
    
    # Print final directory structure
    print("\n" + "=" * 50)
    print("Final directory structure:")
    print(f"  {target_root}/")
    
    # Count directories and files
    dir_count = 0
    file_count = 0
    
    for item in sorted(target_path.iterdir()):
        if item.is_dir():
            dir_count += 1
            videos_in_dir = len([f for f in item.iterdir() if f.is_file()])
            file_count += videos_in_dir
            print(f"  ├── {item.name}/ ({videos_in_dir} videos)")
    
    txt_files = sorted([f for f in target_path.iterdir() if f.is_file() and f.suffix == '.txt'])
    for txt_file in txt_files:
        print(f"  ├── {txt_file.name}")
    
    print(f"\nTotal: {dir_count} directories, {file_count} videos")
    print(f"Random seed used: {random_seed}")
    print("=" * 50)
    print("Sampling completed successfully!")

# Helper function to read the selected files list
def read_selected_files(txt_file):
    """
    Read selected files from txt file
    
    Parameters:
    txt_file: Path to the txt file
    
    Returns:
    List of filenames
    """
    with open(txt_file, 'r', encoding='utf-8') as f:
        # Read all lines and strip whitespace
        filenames = [line.strip() for line in f if line.strip()]
    return filenames

# Helper function to get summary
def get_sampling_summary(summary_file):
    """
    Read sampling summary
    
    Parameters:
    summary_file: Path to the summary file
    
    Returns:
    Dictionary with summary information
    """
    summary = {}
    with open(summary_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            if ':' in line and '/' in line:
                # Parse lines like "Sora: 100 files (100/150)"
                parts = line.strip().split(': ')
                if len(parts) == 2:
                    dir_name = parts[0]
                    files_info = parts[1]
                    summary[dir_name] = files_info
    return summary

if __name__ == "__main__":
    # Configuration
    SOURCE_DIR = "../Data/RealDist/video"           # Directory containing 'fake' and 'real' folders
    TARGET_DIR = "../Data/VQA_videos"  # Output directory
    SAMPLE_SIZE = 100          # Number of videos to sample from each directory
    RANDOM_SEED = 1958         # Fixed random seed for reproducibility
    
    # Run sampling
    sample_videos(SOURCE_DIR, TARGET_DIR, SAMPLE_SIZE, RANDOM_SEED)
    
    # Example of how to read the generated files
    print("\n" + "=" * 50)
    print("Example of reading generated files:")
    
    # Check if files were created
    target_path = Path(TARGET_DIR)
    if target_path.exists():
        # Read summary
        summary_file = target_path / "sampling_summary.txt"
        if summary_file.exists():
            print("\nSampling summary:")
            with open(summary_file, 'r', encoding='utf-8') as f:
                print(f.read())
        
        # Read a sample txt file
        txt_files = [f for f in target_path.iterdir() if f.is_file() and f.name.endswith('_selected.txt')]
        if txt_files:
            sample_file = txt_files[0]
            print(f"\nFirst 5 lines from '{sample_file.name}':")
            filenames = read_selected_files(sample_file)
            for i, filename in enumerate(filenames[:5], 1):
                print(f"  {i}. {filename}")
            if len(filenames) > 5:
                print(f"  ... and {len(filenames)-5} more")