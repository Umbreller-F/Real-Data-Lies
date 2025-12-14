import os
import subprocess
from pathlib import Path
import pandas as pd

# Configuration
BASE_DIR = "../Data/VQA_videos"
SCRIPT_NAME = "evaluate_a_set_of_videos.py"
SCRIPT_PATH = "./data_analysis/DOVER/evaluate_a_set_of_videos.py"
OUTPUT_DIR = "./data_analysis/results"


def main():
    # Get absolute path of base directory
    try:
        base_path = Path(BASE_DIR).resolve(strict=True)
    except FileNotFoundError:
        print(f"Error: Directory not found: {BASE_DIR}")
        print(f"Current working directory: {os.getcwd()}")
        return
    
    if not base_path.is_dir():
        print(f"Error: Not a directory: {base_path}")
        return
    
    # Check if evaluation script exists
    script_path = Path(SCRIPT_PATH)
    if not script_path.exists():
        print(f"Error: Script not found: {script_path}")
        print(f"Trying to locate {SCRIPT_NAME}...")
        # Try to find script in current directory
        if Path(SCRIPT_NAME).exists():
            script_path = Path(SCRIPT_NAME)
            print(f"Found script at: {script_path}")
        else:
            return
    
    # Create output directory if it doesn't exist
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all subdirectories
    subdirs = [d for d in base_path.iterdir() if d.is_dir()]
    # subdirs = [path for path in subdirs if 'Youku' not in str(path)]
    
    if not subdirs:
        print(f"No subdirectories found in: {base_path}")
        return
    
    print(f"Found {len(subdirs)} subdirectories:")
    for i, subdir in enumerate(subdirs, 1):
        print(f"  {i:2d}. {subdir.name}")
    
    # Process each subdirectory
    for i, subdir in enumerate(subdirs, 1):
        subdir_name = subdir.name
        subdir_path = subdir.absolute()
        output_file = output_path / f"VQA/{subdir_name}.csv"
        
        # Ensure VQA directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\n[{i}/{len(subdirs)}] Processing: {subdir_name}")
        print(f"  Input:  {subdir_path}")
        print(f"  Output: {output_file}")
        
        # Build command
        cmd = f"python -W ignore {script_path} --opt data_analysis/DOVER/dover.yml -in {subdir_path} -out {output_file}"
        print(f"  Command: {cmd}")
        
        try:
            # Execute command
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"  ✓ Done")
                if result.stdout.strip():
                    print(f"  Output: {result.stdout.strip()}")
            else:
                print(f"  ✗ Failed (code: {result.returncode})")
                if result.stderr.strip():
                    print(f"  Error: {result.stderr.strip()}")
                    
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print(f"\nAll video processing completed!")
    print(f"Results saved to: {output_path.absolute()}/VQA/")


if __name__ == "__main__":
    main()