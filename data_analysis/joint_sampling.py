import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# # --- Configuration ---
# FILE_PATHS = [
#     'data_analysis/results/joint_sampling/InternVidAES.csv',
#     'data_analysis/results/joint_sampling/K400.csv'
# ]
# OUTPUT_TXT = 'data_analysis/results/joint_sampling/sampled_uniform_10k.txt'
# OUTPUT_IMG = 'data_analysis/results/joint_sampling/distribution_pdf.png'
# TARGET_SIZE = 10000
# SEED = 1958  # Fixed seed for reproducibility

# --- Configuration ---
FILE_PATHS = [
    'data_analysis/results/joint_sampling/InternVidAES_val.csv',
    'data_analysis/results/joint_sampling/K400_val.csv'
]
OUTPUT_TXT = 'data_analysis/results/joint_sampling/sampled_uniform_1k_val.txt'
OUTPUT_IMG = 'data_analysis/results/joint_sampling/distribution_pdf_val.png'
TARGET_SIZE = 1000
SEED = 1958  # Fixed seed for reproducibility

def main():
    # 1. Load and Merge Data
    print(f"Loading data from {len(FILE_PATHS)} sources...")
    df_list = []
    
    for f in FILE_PATHS:
        if os.path.exists(f):
            try:
                # Load specific columns: Index 0 (Path) and Index 3 (Overall Score)
                tmp = pd.read_csv(f, header=None, usecols=[0, 3], names=['path', 'score'], skipinitialspace=True)
                
                # [NEW] Add a column to track the source filename for plotting later
                source_name = os.path.basename(f) # e.g., 'InternVidAES.csv'
                tmp['source'] = source_name
                
                df_list.append(tmp)
                print(f"-> Loaded {len(tmp)} samples from {source_name}")
            except Exception as e:
                print(f"Error reading {f}: {e}")
        else:
            print(f"Warning: File not found {f}")

    if not df_list: return
    
    full_df = pd.concat(df_list, ignore_index=True)
    full_df['score'] = pd.to_numeric(full_df['score'], errors='coerce')
    full_df.dropna(subset=['score'], inplace=True)

    print(f"Total pool size: {len(full_df)} videos.")

    # 2. Inverse Frequency Sampling
    # Binning the data to estimate density (50 bins)
    full_df['bin'] = pd.cut(full_df['score'], bins=50)
    
    # Calculate weight = 1 / (frequency of that score range)
    # observed=False suppresses a future warning in newer pandas versions
    bin_counts = full_df.groupby('bin', observed=False)['path'].transform('count')
    weights = 1.0 / (bin_counts + 1e-6)

    # Sample with fixed random state
    print(f"Sampling {TARGET_SIZE} videos with seed {SEED}...")
    # Handle case if total data < target size (avoid error)
    replace_flag = len(full_df) < TARGET_SIZE
    sampled_df = full_df.sample(n=TARGET_SIZE, weights=weights, replace=replace_flag, random_state=SEED)

    # 3. Plot Probability Density Function (PDF)
    print(f"Generating PDF plot to {OUTPUT_IMG}...")
    plt.figure(figsize=(12, 7)) # Slightly larger for more legend items
    
    # A. Plot Total Original Distribution (Red Filled)
    sns.kdeplot(full_df['score'], fill=True, color='red', alpha=0.15, 
                label='Total Original (Aggregated)', linewidth=0)
    
    # B. [NEW] Plot Individual Source Distributions (Dashed Lines)
    # We loop through unique sources to plot them separately
    palette = sns.color_palette("dark", n_colors=len(df_list)) # distinct colors
    for i, source_name in enumerate(full_df['source'].unique()):
        subset = full_df[full_df['source'] == source_name]
        sns.kdeplot(subset['score'], label=f'Source: {source_name}', 
                    color=palette[i], linestyle='--', linewidth=2)

    # C. Plot Final Sampled Distribution (Blue Filled)
    sns.kdeplot(sampled_df['score'], fill=True, color='blue', alpha=0.25, 
                label=f'Final Sampled (N={TARGET_SIZE})', linewidth=2)

    plt.title(f'Score Distribution Analysis\nSources: {", ".join(full_df["source"].unique())}')
    plt.xlabel('Quality Score (0-100)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 100) # Fix x-axis to score range
    
    # Ensure directory exists before saving
    os.makedirs(os.path.dirname(OUTPUT_IMG), exist_ok=True)
    plt.savefig(OUTPUT_IMG)
    plt.close()

    # 4. Save List
    os.makedirs(os.path.dirname(OUTPUT_TXT), exist_ok=True)
    sampled_df['path'].to_csv(OUTPUT_TXT, index=False, header=False)
    print(f"Done! Path list saved to: {OUTPUT_TXT}")

if __name__ == "__main__":
    main()