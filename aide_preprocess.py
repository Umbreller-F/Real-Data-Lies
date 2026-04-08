import os
import subprocess
from pathlib import Path
import concurrent.futures
import time
import argparse

# Configuration
BASE_DIR = "./assets/aide_image_paths"
SCRIPT_PATH = "./aide_extract_dct_feature.py"

# Patterns for txt files to find
TXT_PATTERNS = ['*.txt']


def find_txt_files(base_dir, target_subdirs=None):
    """Find all txt files in the base directory"""
    txt_files = []
    
    if target_subdirs:
        # Search in specified subdirectories
        for subdir in target_subdirs:
            subdir_path = Path(base_dir) / subdir
            if subdir_path.exists():
                for pattern in TXT_PATTERNS:
                    txt_files.extend(list(subdir_path.rglob(pattern)))
    else:
        # Auto-discover all txt files
        base_path = Path(base_dir)
        if base_path.exists():
            for pattern in TXT_PATTERNS:
                txt_files.extend(list(base_path.rglob(pattern)))
    
    return sorted(txt_files)


def check_already_processed(txt_path):
    """Check if features for this txt file have already been processed"""
    try:
        with open(txt_path, 'r') as f:
            image_paths = [line.strip() for line in f if line.strip()]
        
        if not image_paths:
            return True  # Empty file considered as processed
        
        # Check if first and last feature files exist
        check_paths = [image_paths[0]]
        if len(image_paths) > 1:
            check_paths.append(image_paths[-1])
        
        for img_path in check_paths:
            feature_path = img_path.replace('.jpg', '.pt').replace('video_frames', 'dct_features')
            if not Path(feature_path).exists():
                return False
        
        return True  # All checked feature files exist
    except Exception as e:
        print(f"Warning: Error checking {txt_path}: {e}")
        return False


def process_txt_file(args):
    """Process a single txt file"""
    i, txt_file, total_count = args
    txt_name = txt_file.stem
    txt_dir = txt_file.parent.name
    txt_path = txt_file.absolute()
    
    # Check if already processed
    if check_already_processed(txt_path):
        return i, txt_name, txt_dir, "✓ Skipped (already processed)"
    
    start_time = time.time()
    
    # Build command
    cmd = f"python {SCRIPT_PATH} {txt_path}"
    
    try:
        # Execute command
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            result_message = f"✓ Success - {time.time()-start_time:.1f}s"
        else:
            stderr = result.stderr[-200:] if result.stderr else "No error message"
            result_message = f"✗ Failed (code: {result.returncode}) - {stderr}"
                
    except Exception as e:
        result_message = f"✗ Exception: {str(e)[:50]}"
    
    return i, txt_name, txt_dir, result_message


def main():
    parser = argparse.ArgumentParser(description="Batch DCT feature extraction for multiple txt files")
    parser.add_argument("--base_dir", type=str, default=BASE_DIR, 
                        help="Base directory containing video frames")
    parser.add_argument("--max_workers", type=int, default=4,
                        help="Number of parallel workers (default: 4)")
    args = parser.parse_args()
    
    
    txt_files = find_txt_files(args.base_dir)
    
    if not txt_files:
        print(f"No txt files found in {args.base_dir}")
        return
    
    print(f"Found {len(txt_files)} txt files to process:")
    for i, txt_file in enumerate(txt_files, 1):
        rel_path = txt_file.relative_to(args.base_dir) if args.base_dir in str(txt_file) else txt_file
        print(f"  {i:2d}. {rel_path}")
    
    # Parallel processing
    print(f"\nProcessing {len(txt_files)} txt files with {args.max_workers} workers...")
    
    args_list = [(i, txt_file, len(txt_files)) for i, txt_file in enumerate(txt_files, 1)]
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(process_txt_file, a): a for a in args_list}
        
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            i, txt_name, txt_dir, result_message = future.result()
            completed += 1
            print(f"[{completed}/{len(txt_files)}] {txt_dir}/{txt_name}: {result_message}")
    
    print(f"\nAll txt files processing completed!")


if __name__ == "__main__":
    main()