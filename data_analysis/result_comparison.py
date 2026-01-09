import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def read_and_process_csv(file_path, source_label):
    """Read CSV file and add source label"""
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    df['Source'] = source_label
    df['Dataset'] = df['Dataset'].astype(str).str.strip()
    return df[['Dataset', 'AUROC', 'Source']]

def main():
    # Define your CSV files and their corresponding labels
    # Format: {'file_path': 'legend_label', ...}
    # CSV_CONFIG = {
    #     'results/test/video-classifier/GenVideo-Pika-timesformer-ssv2/RealDist-avg.csv': 'Kinetics-400',
    #     'results/test/video-classifier/GenVideo-Youku-Pika-timesformer-ssv2/RealDist-avg.csv': 'Youku',
    #     'results/test/video-classifier/RealDist-Pika-timesformer-ssv2/RealDist-avg.csv': 'LSVQ',
    #     'results/test/video-classifier/RealDist-I-Pika-timesformer-ssv2/RealDist-avg.csv': 'InternVid-AES',
    # }

    # CSV_CONFIG = {
    #     'results/test/video-classifier/GenVideo-Pika-timesformer-ssv2-NA/RealDist-avg.csv': 'timesformer-ssv2',
    #     'results/test/video-classifier/GenVideo-Pika-videomaev2/RealDist-avg.csv': 'videomaev2',
    #     'results/test/video-classifier/GenVideo-Pika-demamba-NA/RealDist-avg.csv': 'demamba',
    # }

    # CSV_CONFIG = {
    #     'results/test/video-classifier/GenVideo-Pika-demamba/RealDist-avg.csv': 'Kinetics-400',
    #     'results/test/video-classifier/RealDist-I-Pika-demamba/RealDist-avg.csv': 'InternVid-AES',
    # }

    # MODEL = 'demamba'
    # CSV_CONFIG = {
    #     f'results/test/video-classifier/GenVideo-Pika-{MODEL}/RealDist-avg.csv': 'Kinetics-400',
    #     # f'results/test/video-classifier/GenVideo-Youku-Pika-{MODEL}/RealDist-avg.csv': 'Youku',
    #     # f'results/test/video-classifier/RealDist-Pika-{MODEL}/RealDist-avg.csv': 'LSVQ',
    #     f'results/test/video-classifier/RealDist-I-Pika-{MODEL}/RealDist-avg.csv': 'InternVid-AES',
    #     f'results/test/video-classifier/RealDist-U-Pika-{MODEL}/RealDist-avg.csv': 'Uniform',
    # }

    # CSV_CONFIG = {
    #     'results/test/image-classifier/GenVideo-Pika-npr/RealDist-avg.csv': 'npr',
    #     'results/test/image-classifier/GenVideo-Pika-dinov2/RealDist-avg.csv': 'dinov2',
    #     'results/test/image-classifier/GenVideo-Pika-dinov3/RealDist-avg.csv': 'dinov3',
    #     'results/test/video-classifier/GenVideo-Pika-timesformer-ssv2/RealDist-avg.csv': 'timesformer-ssv2',
    #     'results/test/video-classifier/GenVideo-Pika-videomaev2/RealDist-avg.csv': 'videomaev2',
    #     'results/test/video-classifier/GenVideo-Pika-demamba/RealDist-avg.csv': 'demamba',
    #     'results/test/nsg-vd/RealDist-avg.csv': 'nsgvd',
    # }

    # CSV_CONFIG = {
    #     # 'results/test/video-classifier/RealDist-I-Pika-demamba/RealDist-avg.csv': 'InternVidAES',
    #     'results/test/video-classifier/RealDist-I-Pika-demamba-NA/RealDist-avg.csv': 'InternVidAES-NA',
    #     # 'results/test/video-classifier/RealDist-I-Pika-demamba-QM/RealDist-avg.csv': 'InternVidAES-QM',
    #     # 'results/test/video-classifier/RealDist-I-Pika-demamba-NP/RealDist-avg.csv': 'InternVidAES-NP',
    #     # 'results/test/video-classifier/RealDist-I-Pika-demamba-NP-1/RealDist-avg.csv': 'InternVidAES-NP-1',
    #     # 'results/test/video-classifier/GenVideo-Pika-demamba/RealDist-avg.csv': 'K400',
    #     # 'results/test/video-classifier/GenVideo-Pika-demamba-GRL/RealDist-avg.csv': 'K400-GRL',
    #     # 'results/test/video-classifier/GenVideo-Pika-demamba-A3/RealDist-avg.csv': 'K400-A3',
    #     'results/test/video-classifier/GenVideo-Youku-Pika-demamba-NA/RealDist-avg.csv': 'Youku-NA',
    #     'results/test/video-classifier/GenVideo-Pika-demamba-NA/RealDist-avg.csv': 'K400-NA',
    #     'results/test/video-classifier/RealDist-Pika-demamba-NA/RealDist-avg.csv': 'LSVQ-NA',
    #     'results/test/video-classifier/RealDist-O-Pika-demamba-NA/RealDist-avg.csv': 'OpenVidHD-NA',
    # }

    # CSV_CONFIG = {
    #     'results/test/video-classifier/RealDist-I-Pika-videomaev2/RealDist-avg.csv': 'videomaev2',
    #     'results/test/video-classifier/RealDist-I-Pika-videomaev2-NA/RealDist-avg.csv': 'videomaev2-NA',
    #     'results/test/video-classifier/RealDist-I-Pika-videomaev2-LR/RealDist-avg.csv': 'videomaev2-LR',
    #     'results/test/video-classifier/RealDist-I-Pika-videomaev2-NA-LR/RealDist-avg.csv': 'videomaev2-NA-LR',
    #     'results/test/video-classifier/RealDist-I-Pika-videomaev2-A2-LR/RealDist-avg.csv': 'videomaev2-A2-LR',
    #     # 'results/test/video-classifier/RealDist-I-Pika-demamba-QM/RealDist-avg.csv': 'demamba-QM',
    #     # 'results/test/video-classifier/RealDist-I-Pika-demamba-NP/RealDist-avg.csv': 'demamba-NP',
    # }    

    CSV_CONFIG = {
        'results/test/video-classifier/RealDist-I-Pika-demamba-NA/RealDist-avg.csv': 'InternVidAES-NA',
        'results/test/video-classifier/RealDist-I-Pika-demamba-NA-DE1K/RealDist-avg.csv': 'InternVidAES-NA-DE1K',
        'results/test/video-classifier/RealDist-I-Pika-demamba-NA-DE10K/RealDist-avg.csv': 'InternVidAES-NA-DE10K',
    }
    
    # Define the order of datasets (use exact names as in CSV files)
    DATASET_ORDER = [
        'Kinetics-400-Avg',
        'Youku-Avg',
        'MSR-VTT-Avg',
        # 'LSVQ-Avg',
        # 'LSVQ_1080p-Avg', 
        'InternVid-AES-Avg',
        'RealVSR-Avg',
        'GenBuster-200K-benchmark-Avg',
        # 'GenBuster-200K-test-Avg',
        'OpenVidHD-Avg',
    ]

    results_dir = 'data_analysis/results/auc_comparison/'
    save_name = 'I_DE'
    bar_fontsize = 6
    
    # Store all data
    all_data = []
    
    # Read and process all CSV files
    for file_path, label in CSV_CONFIG.items():
        try:
            df = read_and_process_csv(file_path, label)
            all_data.append(df)
            print(f"Processed: {file_path} -> {label}")
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
    
    if not all_data:
        print("No data to plot")
        return
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Get datasets and sources in specified order
    # Filter datasets to only include those in DATASET_ORDER
    valid_datasets = [d for d in DATASET_ORDER if d in combined_df['Dataset'].unique()]
    
    if not valid_datasets:
        print("No matching datasets found. Available datasets:")
        print(combined_df['Dataset'].unique())
        return
    
    # Get sources in the order they appear in CSV_CONFIG
    sources = list(CSV_CONFIG.values())
    
    # Create plot
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    sns.set_palette("deep")
    
    # Create grouped bar positions
    x = np.arange(len(valid_datasets))
    bar_width = 0.8 / len(sources)
    
    # Plot bars for each source
    for i, source in enumerate(sources):
        source_data = combined_df[combined_df['Source'] == source]
        
        # Get AUROC values in dataset order
        values = []
        for dataset in valid_datasets:
            val_data = source_data[source_data['Dataset'] == dataset]['AUROC']
            if not val_data.empty:
                values.append(val_data.values[0])
            else:
                values.append(0)  # Dataset not found in this source
        
        # Calculate bar positions
        positions = x + (i - (len(sources)-1)/2) * bar_width
        
        # Plot bars
        bars = plt.bar(positions, values, width=bar_width, 
                      label=source, edgecolor='black', linewidth=0.8)
        
        # Add value labels
        for bar, value in zip(bars, values):
            if value > 0:
                plt.text(bar.get_x() + bar.get_width()/2, 
                        value + 0.5, f'{value:.2f}', 
                        ha='center', va='bottom', fontsize=bar_fontsize, fontweight='bold')
    
    # Customize plot
    plt.title(f'AUROC Comparison {save_name}', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Dataset', fontsize=12, fontweight='bold')
    plt.ylabel('AUROC (%)', fontsize=12, fontweight='bold')
    # breakpoint()
    plt.xticks(x, valid_datasets, rotation=45, ha='right', fontsize=10)
    plt.ylim(0, 105)
    plt.yticks(fontsize=10)
    plt.legend(title='Method', title_fontsize=12, fontsize=11, 
               bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add grid
    plt.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(results_dir + f'auroc_{save_name}.png', dpi=300, bbox_inches='tight')
    # plt.show()
    
    # Print summary table
    print("\n" + "="*70)
    print("AUROC Summary Table (Ordered by Dataset):")
    print("="*70)
    
    # Reorder pivot table by DATASET_ORDER
    pivot_df = combined_df.pivot(index='Dataset', columns='Source', values='AUROC')
    pivot_df = pivot_df.reindex(valid_datasets)  # Reorder rows
    pivot_df = pivot_df[sources]  # Reorder columns if needed
    
    print(pivot_df.to_string(float_format="%.2f"))
    
    # Add average row
    print("\n" + "-"*70)
    averages = pivot_df.mean()
    avg_df = pd.DataFrame([averages], index=['Average'])
    print(avg_df.to_string(float_format="%.2f"))
    
    # Save data to CSV
    combined_df.to_csv(results_dir + f'combined_auroc_data_{save_name}.csv', index=False)
    pivot_df.to_csv(results_dir + f'auroc_pivot_table_{save_name}.csv')
    print(f"\nCombined data saved to: combined_auroc_data.csv")
    print(f"Pivot table saved to: auroc_pivot_table.csv")
    print(f"Plot saved to: auroc_comparison.png")

if __name__ == "__main__":
    main()