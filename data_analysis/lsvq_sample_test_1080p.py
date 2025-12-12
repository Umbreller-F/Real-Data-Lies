import numpy as np
import pandas as pd
import random
import hashlib


def stratified_sampling_simple(input_file, output_file, n_samples=10000, n_strata=20, random_seed=42):
    """
    Simple version that only uses quality scores for stratification
    """
    # Set seeds
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Read data
    lines = []
    qualities = []
    
    with open(input_file, 'r') as f:
        for line in f:
            if line.strip():
                parts = line.strip().split(',')
                if len(parts) >= 4:
                    lines.append(line.strip())
                    qualities.append(float(parts[3].strip()))
    
    print(f"Total samples: {len(lines)}")
    
    if len(lines) <= n_samples:
        with open(output_file, 'w') as f:
            for line in lines:
                f.write(line + '\n')
        print(f"All {len(lines)} samples written to {output_file}")
        return
    
    # Pair lines with their indices
    data_with_indices = list(enumerate(zip(lines, qualities)))
    
    # Sort by quality
    data_sorted_by_quality = sorted(data_with_indices, key=lambda x: x[1][1])
    
    # Create strata
    strata = []
    stratum_size = len(data_sorted_by_quality) // n_strata
    
    for i in range(n_strata):
        start = i * stratum_size
        end = (i + 1) * stratum_size if i < n_strata - 1 else len(data_sorted_by_quality)
        strata.append(data_sorted_by_quality[start:end])
    
    # Sample from each stratum
    sampled_indices = []
    base_samples = n_samples // n_strata
    extra_samples = n_samples % n_strata
    
    for i, stratum in enumerate(strata):
        stratum_sample_size = base_samples + (1 if i < extra_samples else 0)
        
        if len(stratum) <= stratum_sample_size:
            sampled_indices.extend([item[0] for item in stratum])
        else:
            # Sort by original index for deterministic sampling
            stratum_sorted = sorted(stratum, key=lambda x: x[0])
            selected = random.sample(stratum_sorted, stratum_sample_size)
            sampled_indices.extend([item[0] for item in selected])
    
    # Get sampled lines in original order
    sampled_lines = [lines[i] for i in sorted(sampled_indices)]
    
    # Write output
    with open(output_file, 'w') as f:
        for line in sampled_lines:
            f.write(line + '\n')
    
    # Calculate statistics
    original_qualities = np.array(qualities)
    sampled_qualities = np.array([qualities[i] for i in sampled_indices])
    
    print(f"\nSampling Results:")
    print(f"Samples written: {len(sampled_lines)}")
    print(f"\nQuality Score Comparison:")
    print(f"Original - Mean: {original_qualities.mean():.4f}, Std: {original_qualities.std():.4f}")
    print(f"Sampled  - Mean: {sampled_qualities.mean():.4f}, Std: {sampled_qualities.std():.4f}")
    print(f"Difference - Mean: {abs(original_qualities.mean() - sampled_qualities.mean()):.6f}")
    
    return sampled_lines


# Main execution
if __name__ == "__main__":
    # Configuration
    SAMPLE_COUNT = 1000
    STRATA_COUNT = 20
    RANDOM_SEED = 1958
    lsvq_parts = ['1080p', 'test']
    print(f"Stratified Sampling for Quality Score Distribution")
    for part in lsvq_parts:
        INPUT_FILE = f"data_analysis/LSVQ_labels/labels_{part}.txt"
        OUTPUT_FILE = f"data_analysis/LSVQ_labels/realdist_{part}.txt"
        
        print(f"\n{'='*50}")
        print(f"Input file: {INPUT_FILE}")
        print(f"Output file: {OUTPUT_FILE}")
        print(f"Target samples: {SAMPLE_COUNT}")
        print(f"Number of strata: {STRATA_COUNT}")
        print(f"Random seed: {RANDOM_SEED}")
        print(f"{'='*50}\n")
        
        result = stratified_sampling_simple(
            input_file=INPUT_FILE,
            output_file=OUTPUT_FILE,
            n_samples=SAMPLE_COUNT,
            n_strata=STRATA_COUNT,
            random_seed=RANDOM_SEED
        )
        
        print(f"\n{'='*50}")
        print(f"Sampling completed successfully!")
        print(f"To reproduce exactly, use random_seed={RANDOM_SEED}")
        print(f"Output saved to: {OUTPUT_FILE}")