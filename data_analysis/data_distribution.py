import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_and_combine_files(csv_dir, file_lists_dict):
    """Load CSV files from multiple lists and combine data."""
    all_data = pd.DataFrame()
    
    for list_name, files in file_lists_dict.items():
        for file in files:
            filepath = Path(csv_dir) / file
            if filepath.exists():
                df = pd.read_csv(filepath, header=None, names=["path", "aesthetic", "technical", "overall"])
                df["overall"] = pd.to_numeric(df["overall"], errors='coerce')
                df["list_name"] = list_name
                df["source_file"] = file.replace('.csv', '')
                all_data = pd.concat([all_data, df], ignore_index=True)
            else:
                print(f"Warning: {file} not found")
    
    return all_data

def plot_multiple_densities(data, output_path):
    """Plot density curves for multiple lists."""
    plt.figure(figsize=(10, 6))
    
    # Plot density for each list
    for list_name, group in data.groupby("list_name"):
        scores = group["overall"].dropna()
        if len(scores) > 0:
            sns.kdeplot(data=scores, label=list_name, linewidth=2)
    
    plt.xlabel("Overall Score", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.title("Overall Score Distribution by List", fontsize=14)
    plt.legend(title="List Name")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()

# Configuration
CSV_DIR = "data_analysis/results/VQA"

# Define multiple lists
FILE_LISTS = {
    "AIGV": [
        'Crafter.csv', 'Gen2.csv', 'HotShot.csv', 'Lavie.csv', 
        'ModelScope.csv', 'MoonValley.csv', 'MorphStudio.csv', 
        'Pika.csv', 'SEINE.csv', 'Show_1.csv', 'Sora.csv', 'WildScrape.csv'
    ],
    "InternVid-AES": [
        'InternVid_AES.csv'
    ],
    # "LSVQ": [
    #     'LSVQ.csv'
    # ],
    # "LSVQ-1080p": [
    #     'LSVQ_1080p.csv',
    # ],
    "RealVSR": [
        'RealVSR.csv',
    ],
    "Kinetics-400": [
        'Kinetics-400.csv',
    ],
    # "Youku": [
    #     'Youku.csv',
    # ],
    # "MSR-VTT": [
    #     'MSR-VTT.csv',
    # ],
    "GenBuster-test": [
        'GenBuster-test.csv'
    ],
    "GenBuster-benchmark": [
        'GenBuster-benchmark.csv'
    ],
}

OUTPUT_IMG = "data_analysis/results/density_plots/GB.png"

# Main execution
if __name__ == "__main__":
    # Load data
    combined_data = load_and_combine_files(CSV_DIR, FILE_LISTS)
    
    if combined_data.empty:
        print("No data loaded.")
    else:
        # Plot multiple density curves
        plot_multiple_densities(combined_data, OUTPUT_IMG)
        
        # Print statistics for each list
        print("Statistics by List:")
        for list_name, group in combined_data.groupby("list_name"):
            scores = group["overall"].dropna()
            print(f"\n{list_name}:")
            print(f"  Samples: {len(scores)}")
            print(f"  Mean: {scores.mean():.2f}")
            print(f"  Std: {scores.std():.2f}")