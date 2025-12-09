import os
import subprocess
from pathlib import Path
import pandas as pd

# Configuration
BASE_DIR = "../Data/VQA_videos"
SCRIPT_NAME = "evaluate_a_set_of_videos.py"
SCRIPT_PATH = "./data_analysis/DOVER/evaluate_a_set_of_videos.py"
OUTPUT_DIR = "./data_analysis/results"

def calculate_averages(output_path):
    """Calculate average scores for all CSV files and save to summary file"""
    print(f"\n{'='*60}")
    print("Calculating average scores for all folders")
    print('='*60)
    
    # Find all CSV files
    csv_files = list(output_path.glob("VQA/*.csv"))
    
    if not csv_files:
        print("No CSV files found")
        return
    
    print(f"Found {len(csv_files)} CSV files")
    
    # Prepare list to store summary data
    summary_data = []
    
    for csv_file in csv_files:
        try:
            # Read CSV file
            df = pd.read_csv(csv_file)
            
            # Check if required columns exist
            required_columns = ['aesthetic score', 'technical score', 'overall/final score']
            if not all(col in df.columns for col in required_columns):
                print(f"Warning: {csv_file.name} missing required columns, trying default column names...")
                # Try different column name formats
                column_mapping = {}
                for col in df.columns:
                    col_lower = col.strip().lower()
                    if 'aesthetic' in col_lower:
                        column_mapping[col] = 'aesthetic score'
                    elif 'technical' in col_lower:
                        column_mapping[col] = 'technical score'
                    elif 'overall' in col_lower or 'final' in col_lower:
                        column_mapping[col] = 'overall/final score'
                
                if len(column_mapping) == 3:
                    df = df.rename(columns=column_mapping)
                else:
                    print(f"Skipping {csv_file.name}, unable to identify columns")
                    continue
            
            # Extract folder name (from CSV filename)
            folder_name = csv_file.stem
            
            # Calculate averages
            aesthetic_avg = df['aesthetic score'].mean()
            technical_avg = df['technical score'].mean()
            overall_avg = df['overall/final score'].mean()
            
            # Calculate standard deviations
            aesthetic_std = df['aesthetic score'].std()
            technical_std = df['technical score'].std()
            overall_std = df['overall/final score'].std()
            
            # Count videos
            video_count = len(df)
            
            # Add to summary data
            summary_data.append({
                'folder_name': folder_name,
                'video_count': video_count,
                'aesthetic_mean': aesthetic_avg,
                'aesthetic_std': aesthetic_std,
                'technical_mean': technical_avg,
                'technical_std': technical_std,
                'overall_mean': overall_avg,
                'overall_std': overall_std
            })
            
            print(f"✓ {folder_name:30s} Videos: {video_count:3d} | "
                  f"Aesthetic: {aesthetic_avg:7.3f}±{aesthetic_std:5.3f} | "
                  f"Technical: {technical_avg:7.3f}±{technical_std:5.3f} | "
                  f"Overall: {overall_avg:7.3f}±{overall_std:5.3f}")
                  
        except Exception as e:
            print(f"✗ Error processing {csv_file.name}: {e}")
            continue
    
    if not summary_data:
        print("No CSV files processed successfully")
        return
    
    # Create DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Sort by folder name
    summary_df = summary_df.sort_values('folder_name')
    
    # Calculate overall averages
    total_summary = {
        'folder_name': 'TOTAL_AVERAGE',
        'video_count': summary_df['video_count'].sum(),
        'aesthetic_mean': summary_df['aesthetic_mean'].mean(),
        'aesthetic_std': summary_df['aesthetic_std'].mean(),
        'technical_mean': summary_df['technical_mean'].mean(),
        'technical_std': summary_df['technical_std'].mean(),
        'overall_mean': summary_df['overall_mean'].mean(),
        'overall_std': summary_df['overall_std'].mean()
    }
    
    # Add to DataFrame
    total_df = pd.DataFrame([total_summary])
    summary_df = pd.concat([summary_df, total_df], ignore_index=True)
    
    # Save to CSV file
    summary_csv = output_path / "VQA/vqa_scores_summary.csv"
    summary_df.to_csv(summary_csv, index=False, float_format='%.3f')
    
    print(f"\n{'='*60}")
    print(f"Statistics completed! Results saved to: {summary_csv}")
    print(f"Total folders: {len(summary_data)}")
    print(f"Total videos: {summary_df['video_count'].iloc[-1]}")
    print(f"Overall averages - Aesthetic: {total_summary['aesthetic_mean']:.3f} | "
          f"Technical: {total_summary['technical_mean']:.3f} | "
          f"Overall: {total_summary['overall_mean']:.3f}")
    print('='*60)

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
    
    # Calculate average scores
    calculate_averages(output_path)

if __name__ == "__main__":
    main()