import os
import subprocess
from pathlib import Path
import pandas as pd
import concurrent.futures
import multiprocessing
import time

# Configuration
BASE_DIR = "../Data/RDL/video_frames"
SCRIPT_NAME = "evaluate_a_set_of_videos.py"
SCRIPT_PATH = "./data_analysis/DOVER/evaluate_a_set_of_videos.py"
OUTPUT_DIR = "./data_analysis/VQA_results"
TARGET_SUBDIRS = [
    'fake/Crafter/test', 
    'fake/Gen2/test', 
    # 'fake/HotShot/test', 
    'fake/Lavie/test', 
    'fake/ModelScope/test', 
    'fake/MoonValley/test', 
    'fake/MorphStudio/test', 
    'fake/Pika/val', 
    'fake/SEINE/val', 
    'fake/Show_1/test', 
    'fake/Sora/test', 
    'fake/WildScrape/test',
    'real/Kinetics-400/test', 
    # 'real/LSVQ/test', 
    # 'real/LSVQ_1080p/test', 
    'real/MSR-VTT/test', 
    'real/RealVSR/test', 
    'real/Youku/test',
    'real/InternVid-AES/test',
    'real/OpenVidHD/test',
    'real/Vript/test',
    'real/HD-VG-130M/test',
]


def process_subdir(args):
    """Process a single subdirectory"""
    i, subdir, total_count = args
    subdir_name = subdir.parent.name
    subdir_path = subdir.absolute()
    output_file = Path(OUTPUT_DIR) / f"{subdir_name}.csv"

    if output_file.exists():
        return i, subdir_name, "✓ Skipped (output exists)"
    
    # Ensure VQA directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    
    # Build command
    cmd = f"python -W ignore ./data_analysis/DOVER/evaluate_a_set_of_videos.py --opt data_analysis/DOVER/dover.yml -in {subdir_path} -out {output_file}"
    print(cmd)
    
    result_message = ""
    try:
        # Execute command
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            result_message = f"✓ Success - {time.time()-start_time:.1f}s"
        else:
            result_message = f"✗ Failed (code: {result.returncode}) - {time.time()-start_time:.1f}s"
                
    except Exception as e:
        result_message = f"✗ Exception: {str(e)[:50]}... - {time.time()-start_time:.1f}s"
    
    return i, subdir_name, result_message


def main():
    # Create output directory if it doesn't exist
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all subdirectories
    subdirs = [Path(BASE_DIR) / subdir for subdir in TARGET_SUBDIRS]
    
    print(f"Found {len(subdirs)} subdirectories:")
    for i, subdir in enumerate(subdirs, 1):
        print(f"  {i:2d}. {subdir.parent.name}")

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


if __name__ == "__main__":
    main()