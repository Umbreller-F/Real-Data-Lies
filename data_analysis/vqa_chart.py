import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

# Load the statistics CSV file
df = pd.read_csv('data_analysis/VQA_results/score_statistics.csv')
df.columns = df.columns.str.strip()
df['Model'] = df['Model'].str.strip()

models_to_leave_out = ['LSVQ', 'LSVQ_1080p']
df = df[~df['Model'].isin(models_to_leave_out)]
rename_map = {
    'Youku': 'Youku-mPLUG'
}
df['Model'] = df['Model'].replace(rename_map)
# breakpoint()

# Generate horizontal bar charts for each score type
for score_type in ['Aesthetic', 'Technical', 'Overall']:
    # Sort models by mean score (descending)
    df_sorted = df.sort_values(f'{score_type}_Mean', ascending=False)
    
    # Identify real video models to be excluded
    exclude_models = ['Kinetics-400', 'MSR-VTT', 'RealVSR', 'Youku-mPLUG', 'LSVQ', 'LSVQ_1080p', 'InternVid-AES', 'OpenVidHD', 'Vript', 'HD-VG-130M', 'UltraVideo']
    df_aigv = df_sorted[~df_sorted['Model'].isin(exclude_models)]
    avg_aigv = df_aigv[f'{score_type}_Mean'].mean()
    print(avg_aigv)
    
    # Create figure
    plt.figure(figsize=(12, 10))
    y_pos = np.arange(len(df_sorted))

    # Assign colors based on model type
    colors = []
    for model in df_sorted['Model']:
        if model in exclude_models:
            colors.append('#3498db')  # Blue for excluded (non-AIGV) models
        else:
            colors.append('#e74c3c')  # Red for AI-generated video models

    # Create horizontal bar chart with error bars
    bars = plt.barh(y_pos, df_sorted[f'{score_type}_Mean'], 
                    xerr=df_sorted[f'{score_type}_Std'], 
                    capsize=3, alpha=0.7,
                    color=colors,
                    edgecolor=['darkred' if c=='#e74c3c' else 'darkblue' for c in colors], 
                    linewidth=1)

    # Axis labels and title
    # plt.xlabel(f'{score_type} Score (higher is better)', fontsize=12)
    plt.xlabel(f'DOVER Score', fontsize=12, fontweight='bold')
    plt.ylabel('Real Dataset / AIGV Model', fontsize=12, fontweight='bold')
    # plt.title(f'{score_type} Scores by Model', 
    #         fontsize=14, fontweight='bold', pad=20)

    # Set y-axis ticks to model names
    plt.yticks(y_pos, df_sorted['Model'], fontsize=10)
    plt.gca().invert_yaxis()  # Invert y-axis so highest bar is at the top

    # Add mean value labels at the end of each bar
    for i, (bar, mean_val) in enumerate(zip(bars, df_sorted[f'{score_type}_Mean'])):
        width = bar.get_width()
        plt.text(width + 0.05, bar.get_y() + bar.get_height()/2., 
                f'{mean_val:.2f}', 
                ha='left', va='center', fontsize=12, color='black')

    # Add vertical line for average AIGV score
    plt.axvline(x=avg_aigv, color='red', linestyle='--', 
                alpha=1.0, linewidth=2, label=f'AIGV-Avg: {avg_aigv:.2f}')

    # Create legend patches
    legend_patches = [
        mpatches.Patch(color='#e74c3c', label='AIGV Models'),
        mpatches.Patch(color='#3498db', label='Real Datasets'),
        plt.Line2D([0], [0], color='red', linestyle='--', label=f'AIGV-Avg: {avg_aigv:.2f}')
    ]
    plt.legend(handles=legend_patches, fontsize=9, loc='lower right')

    # Add grid lines
    plt.grid(True, alpha=0.1, axis='x', linestyle='--')

    # Set x-axis limits
    x_max = max(df_sorted[f'{score_type}_Mean']) + df_sorted[f'{score_type}_Std'].max() + 0.3
    plt.xlim(left=3.0, right=x_max)

    # Save the figure
    plt.tight_layout()
    plt.savefig(f'data_analysis/VQA_results/VQA_{score_type}_scores.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'data_analysis/VQA_results/VQA_{score_type}_scores.pdf', dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory