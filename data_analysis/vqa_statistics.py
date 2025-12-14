import pandas as pd
import numpy as np
import os
from pathlib import Path

def calculate_score_stats(csv_dir, output_csv="score_statistics.csv"):
    """
    Calculate mean and standard deviation for three scores in CSV files.
    
    Parameters:
    csv_dir: Directory containing CSV files
    output_csv: Output CSV filename
    """
    csv_dir = Path(csv_dir)
    
    # Check if directory exists
    if not csv_dir.exists():
        print(f"Error: Directory '{csv_dir}' not found.")
        return
    
    # Get all CSV files
    csv_files = list(csv_dir.glob("*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in '{csv_dir}'.")
        return
    
    print(f"Found {len(csv_files)} CSV files:")
    for csv_file in csv_files:
        print(f"  - {csv_file.name}")
    print()
    
    # Store results
    results = []
    
    for csv_file in csv_files:
        if csv_file.name.startswith('score'):
            continue
        try:
            # Read CSV file
            df = pd.read_csv(csv_file, header=None)
            
            # Skip empty files
            if df.empty:
                print(f"Warning: {csv_file.name} is empty, skipping.")
                continue
            
            # Assuming format: path, aesthetic, technical, overall
            # Remove header row if it exists
            if df.iloc[0, 0] == "path":
                df = df.iloc[1:].reset_index(drop=True)
            
            # Convert score columns to numeric
            df[1] = pd.to_numeric(df[1], errors='coerce')
            df[2] = pd.to_numeric(df[2], errors='coerce')
            df[3] = pd.to_numeric(df[3], errors='coerce')
            
            # Drop rows with NaN values
            df_clean = df.dropna(subset=[1, 2, 3])
            
            if len(df_clean) == 0:
                print(f"Warning: {csv_file.name} has no valid scores, skipping.")
                continue
            
            # Calculate mean and std
            aesthetic_mean = df_clean[1].mean()
            aesthetic_std = df_clean[1].std()
            
            technical_mean = df_clean[2].mean()
            technical_std = df_clean[2].std()
            
            overall_mean = df_clean[3].mean()
            overall_std = df_clean[3].std()
            
            # Model name (from CSV filename without extension)
            model_name = csv_file.stem
            
            # Add to results
            results.append({
                "Model": model_name,
                "Aesthetic_Mean": aesthetic_mean,
                "Aesthetic_Std": aesthetic_std,
                "Technical_Mean": technical_mean,
                "Technical_Std": technical_std,
                "Overall_Mean": overall_mean,
                "Overall_Std": overall_std
            })
            
            print(f"Processed {model_name}:")
            print(f"  Aesthetic: Mean={aesthetic_mean:.6f}, Std={aesthetic_std:.6f}")
            print(f"  Technical: Mean={technical_mean:.6f}, Std={technical_std:.6f}")
            print(f"  Overall:   Mean={overall_mean:.6f}, Std={overall_std:.6f}")
            print(f"  Samples:   {len(df_clean)} videos")
            print()
            
        except Exception as e:
            print(f"Error processing {csv_file.name}: {e}")
    
    if not results:
        print("No valid data found in any CSV files.")
        return
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Save to CSV
    output_path = csv_dir / output_csv
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(results_df.to_string(index=False))
    
    return results_df


if __name__ == "__main__":
    # Configuration
    CSV_DIR = "data_analysis/results/VQA"  # Directory containing CSV files
    OUTPUT_FILE = "score_statistics.csv"  # Output filename
    
    # Calculate statistics
    stats_df = calculate_score_stats(CSV_DIR, OUTPUT_FILE)
