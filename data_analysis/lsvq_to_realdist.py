import os
import shutil
from pathlib import Path


def copy_videos_and_save_ids(sampling_txt, source_root, target_dir, id_list_file):
    """
    Copy videos and save their IDs to a text file
    """
    # Create directories
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(os.path.dirname(id_list_file), exist_ok=True)
    
    # Read sampling file
    with open(sampling_txt, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    print(f"Processing {len(lines)} files from {sampling_txt}")
    
    ids = []
    success = 0
    failed = 0
    
    for i, line in enumerate(lines, 1):
        # Extract file path
        if ',' in line:
            file_path = line.split(',')[0].strip()
        else:
            file_path = line.strip()
        
        # Generate new filename
        new_name = file_path.replace('/', '#')
        
        # Full paths
        src = os.path.join(source_root, file_path)
        dst = os.path.join(target_dir, new_name)
        
        # Check source
        if not os.path.exists(src):
            print(f"[{i}] Not found: {file_path}")
            failed += 1
            continue
        
        # Copy file
        try:
            shutil.copy2(src, dst)
            # Save filename
            ids.append(new_name)
            success += 1
            
            if i % 100 == 0:
                print(f"Progress: {i}/{len(lines)}")
        except Exception as e:
            print(f"[{i}] Error: {file_path} - {e}")
            failed += 1
    
    # Save ID list
    with open(id_list_file, 'w') as f:
        for id_name in ids:
            f.write(id_name + '\n')
    
    print(f"\nSummary:")
    print(f"  Success: {success}")
    print(f"  Failed: {failed}")
    print(f"  IDs saved to: {id_list_file}")
    
    return success, failed, ids


def main():
    """Main function to process all splits"""
    SOURCE_ROOT = "../Data/LSVQ"
    
    # Configuration for each split
    configs = {
        'train': {
            'sampling': "data_analysis/LSVQ_labels/realdist_train.txt",
            'target': "../Data/RealDist/video/real/LSVQ",
            'ids': "../Data/RealDist/split/real/LSVQ/train_ids.txt"
        },
        'val': {
            'sampling': "data_analysis/LSVQ_labels/realdist_val.txt",
            'target': "../Data/RealDist/video/real/LSVQ",
            'ids': "../Data/RealDist/split/real/LSVQ/val_ids.txt"
        },
        'test': {
            'sampling': "data_analysis/LSVQ_labels/realdist_test.txt",
            'target': "../Data/RealDist/video/real/LSVQ",
            'ids': "../Data/RealDist/split/real/LSVQ/test_ids.txt"
        },
        '1080p': {
            'sampling': "data_analysis/LSVQ_labels/realdist_1080p.txt",
            'target': "../Data/RealDist/video/real/LSVQ_1080p",
            'ids': "../Data/RealDist/split/real/LSVQ_1080p/test_ids.txt"
        }
    }
    
    print("=" * 60)
    print("RealDist Dataset Preparation")
    print("=" * 60)
    
    # Check source directory
    if not os.path.exists(SOURCE_ROOT):
        print(f"Error: Source directory not found: {SOURCE_ROOT}")
        return
    
    # Process each split
    for split_name, config in configs.items():
        print(f"\n{'='*40}")
        print(f"Processing {split_name} split")
        print(f"{'='*40}")
        
        if not os.path.exists(config['sampling']):
            print(f"Skip: Sampling file not found: {config['sampling']}")
            continue
        
        success, failed, ids = copy_videos_and_save_ids(
            sampling_txt=config['sampling'],
            source_root=SOURCE_ROOT,
            target_dir=config['target'],
            id_list_file=config['ids']
        )
    
    print(f"\n{'='*60}")
    print("All splits processed!")
    print("=" * 60)


if __name__ == "__main__":
    main()