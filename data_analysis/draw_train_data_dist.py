import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Paths configuration
CSV_PATHS = [
    "data_analysis/VQA_results/InternVid-AES.csv",
    "data_analysis/VQA_results/Kinetics-400.csv",
    "data_analysis/VQA_results/Pika.csv",
    "data_analysis/VQA_results/OpenVidHD.csv",
    "data_analysis/VQA_results/Youku.csv",
]
OUTPUT_PATH = "data_analysis/train_dist/train_data_dist.png"
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# Colors for different curves
COLORS = [
    'blue', 
    'green', 
    'red', 
    'orange', 
    'purple', 
    'pink'
]  # Add more colors if needed

# Create plot
plt.figure(figsize=(10, 6))

# Plot separate KDE for each CSV file
for idx, csv_path in enumerate(CSV_PATHS):
    df = pd.read_csv(csv_path, header=None, names=["path", "aesthetic", "technical", "overall"])
    label = os.path.basename(csv_path)  # Use filename as label
    df["overall"] = pd.to_numeric(df["overall"], errors='coerce')
    sns.kdeplot(data=df["overall"], 
                label=label.replace('.csv', '').replace('Youku', 'Youku-mPLUG'), 
                color=COLORS[idx % len(COLORS)], 
                fill=True, 
                alpha=0.3, 
                linewidth=2)

# Add plot labels and legend
plt.xlabel("DOVER Score", fontweight='bold')
plt.ylabel("Density", fontweight='bold')
plt.legend()

# Save plot
plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=300)
plt.savefig(OUTPUT_PATH.replace('png', 'pdf'), dpi=300)
print(f"Plot saved to: {OUTPUT_PATH}")