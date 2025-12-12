import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os
from scipy import stats
from pathlib import Path

def analyze_score_distribution(file_path, save_path, bins=10, output_chart=True):
    """
    Analyze score distribution from a text file
    Args:
        file_path: Path to the text file
        bins: Number of bins for histogram
        output_chart: Whether to output visualization charts
    """
    # Read data
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                # Split each line and get the last column
                parts = line.split(',')
                if len(parts) >= 4:
                    try:
                        score = float(parts[-1].strip())
                        data.append(score)
                    except ValueError:
                        continue
    
    if not data:
        print("No valid data found in the file")
        return
    
    # Convert to numpy array
    scores = np.array(data)
    
    print("=" * 60)
    print("SCORE DISTRIBUTION ANALYSIS")
    print("=" * 60)
    print(f"Total samples: {len(scores)}")
    print(f"Score range: [{scores.min():.2f}, {scores.max():.2f}]")
    print(f"Mean: {scores.mean():.4f}")
    print(f"Median: {np.median(scores):.4f}")
    print(f"Standard deviation: {scores.std():.4f}")
    print(f"Variance: {scores.var():.4f}")
    print(f"25th percentile: {np.percentile(scores, 25):.4f}")
    print(f"75th percentile: {np.percentile(scores, 75):.4f}")
    print(f"IQR (Q3-Q1): {np.percentile(scores, 75) - np.percentile(scores, 25):.4f}")
    
    # Method 1: Uniform bin statistics
    print("\n" + "=" * 60)
    print("METHOD 1: UNIFORM BIN STATISTICS")
    print("=" * 60)
    
    hist, bin_edges = np.histogram(scores, bins=bins)
    
    print(f"\nDividing range [{scores.min():.2f}, {scores.max():.2f}] into {bins} bins:")
    for i in range(len(hist)):
        lower = bin_edges[i]
        upper = bin_edges[i+1]
        count = hist[i]
        percentage = (count / len(scores)) * 100
        print(f"Bin [{lower:.2f}, {upper:.2f}): {count} samples ({percentage:.1f}%)")
    
    # Method 2: Fixed threshold statistics
    print("\n" + "=" * 60)
    print("METHOD 2: FIXED THRESHOLD STATISTICS")
    print("=" * 60)
    
    thresholds = [0, 50, 60, 70, 80, 90, 100]
    threshold_counts = defaultdict(int)
    
    for score in scores:
        for i in range(len(thresholds)-1):
            if thresholds[i] <= score < thresholds[i+1]:
                threshold_counts[f"{thresholds[i]}-{thresholds[i+1]}"] += 1
                break
        else:
            if score >= thresholds[-1]:
                threshold_counts[f"{thresholds[-1]}+"] += 1
    
    print("\nDistribution by score thresholds:")
    for key in sorted(threshold_counts.keys()):
        count = threshold_counts[key]
        percentage = (count / len(scores)) * 100
        print(f"{key}: {count} samples ({percentage:.1f}%)")
    
    # Method 3: Decile statistics
    print("\n" + "=" * 60)
    print("METHOD 3: DECILE STATISTICS")
    print("=" * 60)
    
    deciles = np.percentile(scores, [i*10 for i in range(1, 10)])
    print("Decile values:")
    for i, value in enumerate(deciles, 1):
        print(f"{i*10}th percentile: {value:.4f}")
    
    # Method 4: Score category analysis
    print("\n" + "=" * 60)
    print("METHOD 4: SCORE CATEGORY ANALYSIS")
    print("=" * 60)
    
    categories = {
        "Very Low (0-40)": ((scores >= 0) & (scores < 40)).sum(),
        "Low (40-60)": ((scores >= 40) & (scores < 60)).sum(),
        "Medium (60-70)": ((scores >= 60) & (scores < 70)).sum(),
        "Good (70-80)": ((scores >= 70) & (scores < 80)).sum(),
        "High (80-90)": ((scores >= 80) & (scores < 90)).sum(),
        "Excellent (90-100)": ((scores >= 90) & (scores <= 100)).sum(),
    }
    
    for category, count in categories.items():
        if count > 0:
            percentage = (count / len(scores)) * 100
            print(f"{category}: {count} samples ({percentage:.1f}%)")
    
    # Output CSV file with statistics
    output_csv = os.path.join(save_path, f"{Path(file_path).stem}_stats.csv")
    with open(output_csv, 'w', encoding='utf-8') as f:
        f.write("Statistic,Value\n")
        f.write(f"Total_samples,{len(scores)}\n")
        f.write(f"Minimum,{scores.min():.4f}\n")
        f.write(f"Maximum,{scores.max():.4f}\n")
        f.write(f"Range,{scores.max() - scores.min():.4f}\n")
        f.write(f"Mean,{scores.mean():.4f}\n")
        f.write(f"Median,{np.median(scores):.4f}\n")
        f.write(f"Mode,{stats.mode(scores, keepdims=True).mode[0]:.4f}\n")
        f.write(f"Std_deviation,{scores.std():.4f}\n")
        f.write(f"Variance,{scores.var():.4f}\n")
        f.write(f"25th_percentile,{np.percentile(scores, 25):.4f}\n")
        f.write(f"75th_percentile,{np.percentile(scores, 75):.4f}\n")
        f.write(f"IQR,{np.percentile(scores, 75) - np.percentile(scores, 25):.4f}\n")
        f.write(f"Skewness,{stats.skew(scores):.4f}\n")
        f.write(f"Kurtosis,{stats.kurtosis(scores):.4f}\n")
        
        # Add bin statistics
        f.write("\nBin Statistics\n")
        f.write("Bin,Count,Percentage\n")
        for i in range(len(hist)):
            lower = bin_edges[i]
            upper = bin_edges[i+1]
            count = hist[i]
            percentage = (count / len(scores)) * 100
            f.write(f"{lower:.2f}-{upper:.2f},{count},{percentage:.2f}\n")
        
        # Add threshold statistics
        f.write("\nThreshold Statistics\n")
        f.write("Threshold,Count,Percentage\n")
        for key in sorted(threshold_counts.keys()):
            count = threshold_counts[key]
            percentage = (count / len(scores)) * 100
            f.write(f"{key},{count},{percentage:.2f}\n")
    
    print(f"\nDetailed statistics saved to: {output_csv}")
    
    # Generate visualizations if requested
    if output_chart:
        try:
            create_visualizations(scores, file_path)
        except Exception as e:
            print(f"\nError creating visualization: {e}")
            print("Continuing with text output only...")

def create_visualizations(scores, file_path):
    """Create visualization charts for score distribution"""
    # Create 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Histogram
    axes[0, 0].hist(scores, bins=20, edgecolor='black', alpha=0.7, color='skyblue')
    axes[0, 0].set_xlabel('Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Score Distribution Histogram')
    axes[0, 0].grid(True, alpha=0.3, linestyle='--')
    
    # Add mean and median lines
    mean_val = np.mean(scores)
    median_val = np.median(scores)
    axes[0, 0].axvline(mean_val, color='red', linestyle='--', alpha=0.7, 
                      label=f'Mean: {mean_val:.2f}')
    axes[0, 0].axvline(median_val, color='green', linestyle='--', alpha=0.7, 
                      label=f'Median: {median_val:.2f}')
    axes[0, 0].legend()
    
    # 2. Box plot
    boxplot = axes[0, 1].boxplot(scores, patch_artist=True, 
                                 boxprops=dict(facecolor='lightgreen', alpha=0.7),
                                 medianprops=dict(color='red', linewidth=2),
                                 whiskerprops=dict(color='black', linestyle='--'),
                                 capprops=dict(color='black'),
                                 flierprops=dict(marker='o', color='red', alpha=0.5))
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_title('Score Box Plot')
    axes[0, 1].grid(True, alpha=0.3, linestyle='--')
    
    # Add statistics on the side
    stats_text = f"Min: {np.min(scores):.2f}\n"
    stats_text += f"Q1: {np.percentile(scores, 25):.2f}\n"
    stats_text += f"Median: {median_val:.2f}\n"
    stats_text += f"Q3: {np.percentile(scores, 75):.2f}\n"
    stats_text += f"Max: {np.max(scores):.2f}\n"
    stats_text += f"IQR: {np.percentile(scores, 75) - np.percentile(scores, 25):.2f}"
    
    axes[0, 1].text(1.15, 0.5, stats_text, transform=axes[0, 1].transAxes,
                   verticalalignment='center', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 3. Density plot
    try:
        density = stats.gaussian_kde(scores)
        xs = np.linspace(scores.min() - 1, scores.max() + 1, 200)
        axes[1, 0].plot(xs, density(xs), color='darkorange', linewidth=2)
        axes[1, 0].fill_between(xs, density(xs), alpha=0.5, color='orange')
        axes[1, 0].set_xlabel('Score')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title('Score Density Plot')
        axes[1, 0].grid(True, alpha=0.3, linestyle='--')
    except:
        # Fallback to histogram if KDE fails
        axes[1, 0].hist(scores, bins=30, density=True, alpha=0.7, 
                       color='orange', edgecolor='black')
        axes[1, 0].set_xlabel('Score')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title('Normalized Histogram (Density)')
        axes[1, 0].grid(True, alpha=0.3, linestyle='--')
    
    # 4. Cumulative distribution plot
    sorted_scores = np.sort(scores)
    y = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
    axes[1, 1].plot(sorted_scores, y, marker='.', linestyle='-', 
                   color='purple', alpha=0.7, markersize=3, linewidth=1)
    axes[1, 1].set_xlabel('Score')
    axes[1, 1].set_ylabel('Cumulative Probability')
    axes[1, 0].set_title('Probability Density Function')
    axes[1, 1].set_title('Cumulative Distribution Function')
    axes[1, 1].grid(True, alpha=0.3, linestyle='--')
    
    # Add reference lines at 25%, 50%, 75%
    for percentile, color, label in [(25, 'blue', '25%'), (50, 'green', '50%'), (75, 'red', '75%')]:
        value = np.percentile(scores, percentile)
        axes[1, 1].axvline(value, color=color, linestyle='--', alpha=0.5)
        axes[1, 1].text(value, 0.1, f'{percentile}%: {value:.1f}', 
                       rotation=90, fontsize=8, color=color)
    
    plt.suptitle('Score Distribution Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the figure
    output_img = os.path.join(save_path, f"{Path(file_path).stem}_distribution.png")
    plt.savefig(output_img, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Distribution chart saved to: {output_img}")
    plt.show()

def simple_score_stats(file_path):
    """
    Simple version of score statistics
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(',')
                if len(parts) >= 4:
                    try:
                        score = float(parts[-1].strip())
                        data.append(score)
                    except ValueError:
                        continue
    
    if not data:
        print("No valid data found in the file")
        return
    
    scores = np.array(data)
    
    print("\n" + "=" * 60)
    print("QUICK STATISTICS SUMMARY")
    print("=" * 60)
    print(f"Total samples: {len(scores)}")
    print(f"Score range: {scores.min():.2f} ~ {scores.max():.2f}")
    print(f"Mean: {scores.mean():.2f}")
    print(f"Median: {np.median(scores):.2f}")
    print(f"Standard deviation: {scores.std():.2f}")
    
    # Statistics by 10-point intervals
    print("\nDistribution by 10-point intervals:")
    min_score = int(np.floor(scores.min() / 10) * 10)
    max_score = int(np.ceil(scores.max() / 10) * 10)
    
    for lower in range(min_score, max_score, 10):
        upper = lower + 10
        if upper > max_score:
            break
        
        count = np.sum((scores >= lower) & (scores < upper))
        if count > 0:
            percentage = (count / len(scores)) * 100
            print(f"[{lower:3d}, {upper:3d}): {count:3d} samples ({percentage:5.1f}%)")
    
    # Count samples in different quality ranges
    print("\nQuality assessment:")
    
    quality_ranges = [
        ("Poor (0-50)", (scores >= 0) & (scores < 50)),
        ("Fair (50-60)", (scores >= 50) & (scores < 60)),
        ("Average (60-70)", (scores >= 60) & (scores < 70)),
        ("Good (70-80)", (scores >= 70) & (scores < 80)),
        ("Very Good (80-90)", (scores >= 80) & (scores < 90)),
        ("Excellent (90-100)", (scores >= 90) & (scores <= 100)),
    ]
    
    for label, condition in quality_ranges:
        count = np.sum(condition)
        if count > 0:
            percentage = (count / len(scores)) * 100
            print(f"{label}: {count:3d} samples ({percentage:5.1f}%)")
    
    # Additional statistics
    print(f"\nSamples >= 80 (High Quality): {np.sum(scores >= 80)} ({np.sum(scores >= 80)/len(scores)*100:.1f}%)")
    print(f"Samples < 60 (Low Quality): {np.sum(scores < 60)} ({np.sum(scores < 60)/len(scores)*100:.1f}%)")
    
    # Check for outliers (using IQR method)
    Q1 = np.percentile(scores, 25)
    Q3 = np.percentile(scores, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = scores[(scores < lower_bound) | (scores > upper_bound)]
    
    if len(outliers) > 0:
        print(f"\nPotential outliers (IQR method): {len(outliers)} samples")
        print(f"Outlier range: [{outliers.min():.2f}, {outliers.max():.2f}]")

# def batch_analyze_files(file_patterns, output_summary="summary_report.txt"):
#     """
#     Analyze multiple files and generate a summary report
#     Args:
#         file_patterns: List of file paths or patterns
#         output_summary: Path for summary report
#     """
#     all_stats = []
    
#     for file_path in file_patterns:
#         if os.path.exists(file_path):
#             print(f"\nAnalyzing: {file_path}")
#             print("-" * 40)
            
#             data = []
#             with open(file_path, 'r', encoding='utf-8') as f:
#                 for line in f:
#                     line = line.strip()
#                     if line:
#                         parts = line.split(',')
#                         if len(parts) >= 4:
#                             try:
#                                 score = float(parts[-1].strip())
#                                 data.append(score)
#                             except ValueError:
#                                 continue
            
#             if data:
#                 scores = np.array(data)
#                 stats = {
#                     'file': file_path,
#                     'count': len(scores),
#                     'min': scores.min(),
#                     'max': scores.max(),
#                     'mean': scores.mean(),
#                     'median': np.median(scores),
#                     'std': scores.std(),
#                     'q1': np.percentile(scores, 25),
#                     'q3': np.percentile(scores, 75),
#                 }
#                 all_stats.append(stats)
                
#                 print(f"  Samples: {len(scores)}")
#                 print(f"  Mean: {scores.mean():.2f}")
#                 print(f"  Range: {scores.min():.2f} - {scores.max():.2f}")
#             else:
#                 print(f"  No valid data in {file_path}")
#         else:
#             print(f"  File not found: {file_path}")
    
#     # Write summary report
#     if all_stats:
#         with open(output_summary, 'w') as f:
#             f.write("SCORE ANALYSIS SUMMARY REPORT\n")
#             f.write("=" * 50 + "\n\n")
            
#             for stats in all_stats:
#                 f.write(f"File: {stats['file']}\n")
#                 f.write(f"  Samples: {stats['count']}\n")
#                 f.write(f"  Min: {stats['min']:.2f}\n")
#                 f.write(f"  Max: {stats['max']:.2f}\n")
#                 f.write(f"  Mean: {stats['mean']:.2f}\n")
#                 f.write(f"  Median: {stats['median']:.2f}\n")
#                 f.write(f"  Std Dev: {stats['std']:.2f}\n")
#                 f.write(f"  Q1: {stats['q1']:.2f}\n")
#                 f.write(f"  Q3: {stats['q3']:.2f}\n")
#                 f.write(f"  IQR: {stats['q3'] - stats['q1']:.2f}\n")
#                 f.write("-" * 40 + "\n")
        
#         print(f"\nSummary report saved to: {output_summary}")

# Main execution
if __name__ == "__main__":
    # Set your file path here
    txt_files = ['labels_1080p.txt', 'labels_test.txt', 'labels.txt', 'train_labels.txt', 'realdist_1080p.txt', 'realdist_test.txt', 'realdist_train.txt', 'realdist_val.txt']
    for txt_file in txt_files:
        file_path = f"data_analysis/LSVQ_labels/{txt_file}"  # Change this to your actual file path
        save_path = "data_analysis/results/LSVQ-statistics"
        
        print("=" * 60)
        print("SCORE DISTRIBUTION ANALYZER")
        print("=" * 60)
        print(f"Analyzing file: {file_path}")
        print(f"File size: {os.path.getsize(file_path)} bytes")
        
        # Run complete analysis
        analyze_score_distribution(file_path, save_path, bins=10, output_chart=True)
        
        # Run simple statistics
        simple_score_stats(file_path)
        
        print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)