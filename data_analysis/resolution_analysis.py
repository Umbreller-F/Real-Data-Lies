import cv2
from pathlib import Path
from collections import defaultdict

def analyze_resolutions(root_path):
    """Analyze video resolutions from frame images."""
    
    resolution_stats = defaultdict(lambda: defaultdict(int))
    
    for category in ['real', 'fake']:
        category_path = Path(root_path) / category
        
        # Get all subdirectories
        for subdir in category_path.iterdir():
            if not subdir.is_dir():
                continue
                
            # Find test directories
            test_dirs = list(subdir.rglob('test'))
            
            for test_dir in test_dirs:
                # Get all video frame directories
                for video_dir in test_dir.iterdir():
                    if not video_dir.is_dir():
                        continue
                    
                    # Find first frame
                    first_frame = None
                    for frame_pattern in ['frame0001.jpg', 'frame_0001.jpg', 'frame_000001.jpg']:
                        potential_frame = video_dir / frame_pattern
                        if potential_frame.exists():
                            first_frame = potential_frame
                            break
                    
                    if not first_frame:
                        # Try to find any frame image
                        frame_files = list(video_dir.glob('frame*.jpg'))
                        if frame_files:
                            first_frame = sorted(frame_files)[0]
                    
                    if first_frame:
                        # Read image resolution
                        img = cv2.imread(str(first_frame))
                        if img is not None:
                            height, width = img.shape[:2]
                            res_key = f"{width}x{height}"
                            resolution_stats[category][res_key] += 1
                            resolution_stats[subdir.name][res_key] += 1
    
    return resolution_stats

def save_statistics(stats, output_file):
    """Save resolution statistics to a text file."""
    
    with open(output_file, 'w') as f:
        f.write("=" * 50 + "\n")
        f.write("RESOLUTION STATISTICS\n")
        f.write("=" * 50 + "\n\n")
        
        # Overall statistics
        for category in ['real', 'fake']:
            if category in stats:
                f.write(f"{category.upper()} Category:\n")
                f.write("-" * 30 + "\n")
                total_videos = sum(stats[category].values())
                f.write(f"Total videos: {total_videos}\n")
                
                for res, count in sorted(stats[category].items(), 
                                       key=lambda x: x[1], reverse=True):
                    percentage = (count / total_videos * 100) if total_videos > 0 else 0
                    f.write(f"  {res}: {count} ({percentage:.1f}%)\n")
                f.write("\n")
        
        # Per-dataset statistics
        f.write("=" * 50 + "\n")
        f.write("PER-DATASET STATISTICS\n")
        f.write("=" * 50 + "\n\n")
        
        datasets = {k: v for k, v in stats.items() 
                   if k not in ['real', 'fake']}
        
        for dataset, resolutions in sorted(datasets.items()):
            f.write(f"{dataset}:\n")
            f.write("-" * 20 + "\n")
            total = sum(resolutions.values())
            
            for res, count in sorted(resolutions.items(), 
                                   key=lambda x: x[1], reverse=True):
                percentage = (count / total * 100) if total > 0 else 0
                f.write(f"  {res}: {count} ({percentage:.1f}%)\n")
            f.write("\n")

if __name__ == "__main__":
    root_dir = "../Data/RealDist/video_frames"
    output_path = "data_analysis/results/resolution/resolution_report.txt"
    
    print(f"Analyzing resolutions in: {root_dir}")
    statistics = analyze_resolutions(root_dir)
    
    save_statistics(statistics, output_path)
    print(f"Report saved to: {output_path}")