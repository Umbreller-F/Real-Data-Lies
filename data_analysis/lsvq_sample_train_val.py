import numpy as np
import pandas as pd
import random


def stratified_sampling_with_exclusion(input_file, output_file, n_samples=10000, n_strata=20, 
                                      random_seed=42, exclude_lines=None, exclude_qualities=None):
    """
    Stratified sampling that excludes specified samples
    
    Args:
        input_file: Input txt file path
        output_file: Output txt file path
        n_samples: Number of samples to extract
        n_strata: Number of strata/bins
        random_seed: Random seed for reproducibility
        exclude_lines: List of lines to exclude
        exclude_qualities: Corresponding qualities of lines to exclude
    """
    # Set seeds
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Read data
    all_lines = []
    all_qualities = []
    
    with open(input_file, 'r') as f:
        for line in f:
            if line.strip():
                parts = line.strip().split(',')
                if len(parts) >= 4:
                    all_lines.append(line.strip())
                    all_qualities.append(float(parts[3].strip()))
    
    print(f"Total samples: {len(all_lines)}")
    
    # Exclude training samples if provided
    if exclude_lines is not None and exclude_qualities is not None:
        # Create sets for efficient lookup
        exclude_set = set(exclude_lines)
        
        # Filter out excluded lines
        filtered_lines = []
        filtered_qualities = []
        
        for line, quality in zip(all_lines, all_qualities):
            if line not in exclude_set:
                filtered_lines.append(line)
                filtered_qualities.append(quality)
        
        print(f"After excluding {len(exclude_set)} training samples: {len(filtered_lines)} available")
        
        if len(filtered_lines) == 0:
            print("Error: No samples available after exclusion")
            return []
            
        lines = filtered_lines
        qualities = filtered_qualities
    else:
        lines = all_lines
        qualities = all_qualities
    
    # Check if enough samples
    if len(lines) <= n_samples:
        with open(output_file, 'w') as f:
            for line in lines:
                f.write(line + '\n')
        print(f"All {len(lines)} samples written to {output_file}")
        return lines
    
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


def sample_train_val_no_overlap(input_file, train_output, val_output, 
                               train_samples=10000, val_samples=1000,
                               n_strata=20, random_seed_train=1958, random_seed_val=2025):
    """
    Sample training and validation sets with no overlap
    """
    print(f"Sampling Training and Validation Sets with No Overlap")
    print(f"{'='*60}")
    print(f"Input file: {input_file}")
    print(f"Training output: {train_output}")
    print(f"Validation output: {val_output}")
    print(f"Training samples: {train_samples}")
    print(f"Validation samples: {val_samples}")
    print(f"Training random seed: {random_seed_train}")
    print(f"Validation random seed: {random_seed_val}")
    print(f"{'='*60}\n")
    
    # Step 1: Sample training set
    print("STEP 1: Sampling training set...")
    train_lines = stratified_sampling_with_exclusion(
        input_file=input_file,
        output_file=train_output,
        n_samples=train_samples,
        n_strata=n_strata,
        random_seed=random_seed_train
    )
    
    # Extract training qualities for statistics
    train_qualities = []
    for line in train_lines:
        parts = line.split(',')
        train_qualities.append(float(parts[3].strip()))
    
    print(f"\n{'='*60}")
    print("STEP 2: Sampling validation set (excluding training samples)...")
    
    # Step 2: Sample validation set excluding training samples
    val_lines = stratified_sampling_with_exclusion(
        input_file=input_file,
        output_file=val_output,
        n_samples=val_samples,
        n_strata=n_strata,
        random_seed=random_seed_val,
        exclude_lines=train_lines,
        exclude_qualities=train_qualities
    )
    
    # Extract validation qualities
    val_qualities = []
    for line in val_lines:
        parts = line.split(',')
        val_qualities.append(float(parts[3].strip()))
    
    # Check for overlap
    train_set = set(train_lines)
    val_set = set(val_lines)
    overlap = train_set.intersection(val_set)
    
    print(f"\n{'='*60}")
    print("OVERLAP CHECK:")
    print(f"Training set size: {len(train_lines)}")
    print(f"Validation set size: {len(val_lines)}")
    print(f"Overlap count: {len(overlap)}")
    
    if len(overlap) == 0:
        print("✓ SUCCESS: No overlap between training and validation sets")
    else:
        print(f"✗ ERROR: Found {len(overlap)} overlapping samples")
        for i, line in enumerate(list(overlap)[:5], 1):
            print(f"  Overlap {i}: {line[:50]}...")
    
    # Overall statistics
    print(f"\n{'='*60}")
    print("OVERALL STATISTICS:")
    
    # Read original file for comparison
    all_qualities = []
    with open(input_file, 'r') as f:
        for line in f:
            if line.strip():
                parts = line.strip().split(',')
                if len(parts) >= 4:
                    all_qualities.append(float(parts[3].strip()))
    
    print(f"Original dataset: {len(all_qualities)} samples")
    print(f"Training set: {len(train_lines)} samples ({len(train_lines)/len(all_qualities)*100:.1f}%)")
    print(f"Validation set: {len(val_lines)} samples ({len(val_lines)/len(all_qualities)*100:.1f}%)")
    print(f"Total sampled: {len(train_lines) + len(val_lines)} samples")
    print(f"Remaining: {len(all_qualities) - len(train_lines) - len(val_lines)} samples")
    
    print(f"\nQuality Statistics:")
    print(f"{'Dataset':<12} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print(f"{'-'*52}")
    print(f"{'Original':<12} {np.mean(all_qualities):<10.4f} {np.std(all_qualities):<10.4f} "
          f"{min(all_qualities):<10.4f} {max(all_qualities):<10.4f}")
    print(f"{'Training':<12} {np.mean(train_qualities):<10.4f} {np.std(train_qualities):<10.4f} "
          f"{min(train_qualities):<10.4f} {max(train_qualities):<10.4f}")
    print(f"{'Validation':<12} {np.mean(val_qualities):<10.4f} {np.std(val_qualities):<10.4f} "
          f"{min(val_qualities):<10.4f} {max(val_qualities):<10.4f}")
    
    return train_lines, val_lines


# Main execution
if __name__ == "__main__":
    # Configuration
    INPUT_FILE = "data_analysis/LSVQ_labels/train_labels.txt"
    TRAIN_OUTPUT = "data_analysis/LSVQ_labels/realdist_train.txt"
    VAL_OUTPUT = "data_analysis/LSVQ_labels/realdist_val.txt"
    TRAIN_SAMPLES = 10000
    VAL_SAMPLES = 1000
    STRATA_COUNT = 20
    RANDOM_SEED_TRAIN = 1958
    RANDOM_SEED_VAL = 2025
    
    # Execute sampling
    train_lines, val_lines = sample_train_val_no_overlap(
        input_file=INPUT_FILE,
        train_output=TRAIN_OUTPUT,
        val_output=VAL_OUTPUT,
        train_samples=TRAIN_SAMPLES,
        val_samples=VAL_SAMPLES,
        n_strata=STRATA_COUNT,
        random_seed_train=RANDOM_SEED_TRAIN,
        random_seed_val=RANDOM_SEED_VAL
    )
    
    print(f"\n{'='*60}")
    print("SAMPLING COMPLETED SUCCESSFULLY!")
    print(f"Training set saved to: {TRAIN_OUTPUT}")
    print(f"Validation set saved to: {VAL_OUTPUT}")
    print(f"No overlap guaranteed between the two sets")
    print(f"{'='*60}")