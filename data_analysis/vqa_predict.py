import os
import subprocess
from pathlib import Path
import pandas as pd
import concurrent.futures
import multiprocessing
import time

# Configuration
BASE_DIR = "../Data/VQA_videos"
SCRIPT_NAME = "evaluate_a_set_of_videos.py"
SCRIPT_PATH = "./data_analysis/DOVER/evaluate_a_set_of_videos.py"
OUTPUT_DIR = "./data_analysis/results"


def process_subdir(args):
    """Process a single subdirectory"""
    i, subdir, total_count = args
    subdir_name = subdir.name
    subdir_path = subdir.absolute()
    output_file = Path(OUTPUT_DIR) / f"VQA/{subdir_name}.csv"

    if output_file.exists():
        return i, subdir_name, "✓ Skipped (output exists)"
    
    # Ensure VQA directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    
    # Build command
    cmd = f"python -W ignore ./data_analysis/DOVER/evaluate_a_set_of_videos.py --opt data_analysis/DOVER/dover.yml -in {subdir_path} -out {output_file}"
    
    result_message = ""
    try:
        # Execute command
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            result_message = f"✓ Success - {time.time()-start_time:.1f}s"
            # if result.stdout.strip():
            #     result_message += f" | Output: {result.stdout.strip()[:50]}..."
        else:
            result_message = f"✗ Failed (code: {result.returncode}) - {time.time()-start_time:.1f}s"
            # if result.stderr.strip():
            #     result_message += f" | Error: {result.stderr.strip()[:50]}..."
                
    except Exception as e:
        result_message = f"✗ Exception: {str(e)[:50]}... - {time.time()-start_time:.1f}s"
    
    return i, subdir_name, result_message


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
    
    '''# Process each subdirectory
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
            print(f"  ✗ Error: {e}")'''

    # Replace the sequential processing loop with parallel execution
    print(f"\nProcessing {len(subdirs)} directories in parallel...")

    # Determine number of parallel processes (use CPU count or limit as needed)
    max_workers = 8
    print(f"Using {max_workers} parallel workers")

    # Prepare arguments for parallel processing
    args_list = [(i, subdir, len(subdirs)) for i, subdir in enumerate(subdirs, 1)]

    # Execute in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_subdir, args): args for args in args_list}
        
        # Monitor progress
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            i, subdir_name, result_message = future.result()
            completed += 1
            print(f"[{completed}/{len(subdirs)}] {subdir_name}: {result_message}")
    
    print(f"\nAll video processing completed!")
    print(f"Results saved to: {output_path.absolute()}/VQA/")


if __name__ == "__main__":
    main()