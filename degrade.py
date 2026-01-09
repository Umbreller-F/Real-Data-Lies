import os
import subprocess
import random
from pathlib import Path

def degrade_videos(input_dir, output_dir, crf_range=(35, 50), downscale_prob=0.5):
    """
    Degrade all mp4 and webm videos in input_dir and save to output_dir.
    
    Args:
        input_dir: Directory containing input videos
        output_dir: Directory to save degraded videos
        crf_range: Tuple of (min_crf, max_crf) for quality degradation
        downscale_prob: Probability of applying 2x downscaling (0.0 to 1.0)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Supported video extensions
    video_exts = {'.mp4', '.webm', '.MP4', '.WEBM'}
    
    for video_file in input_path.iterdir():
        if video_file.suffix in video_exts:
            # Generate output path
            output_file = output_path / video_file.name
            
            # Random degradation parameters
            crf = random.randint(*crf_range)
            scale_filter = "-vf scale=iw/2:ih/2" if random.random() < downscale_prob else ""
            
            # FFmpeg command
            cmd = [
                'ffmpeg', '-y', '-i', str(video_file),
                '-c:v', 'libx264', '-crf', str(crf),
                '-c:a', 'copy', output_file
            ]
            
            # Add scale filter if needed
            if scale_filter:
                cmd.insert(-1, '-vf')
                cmd.insert(-1, 'scale=iw/2:ih/2')
            
            # Execute command
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"Degraded: {video_file.name} -> CRF: {crf} {'(downscaled)' if scale_filter else ''}")

if __name__ == "__main__":
    # Configuration
    # INPUT_DIR = "../Data/data_for_joint_sampling/OpenVidHD"  # Modify this
    # OUTPUT_DIR = "../Data/degraded_videos/OpenVidHD"  # Modify this
    INPUT_DIR = "../Data/data_for_joint_sampling/Pika"
    OUTPUT_DIR = "../Data/degraded_videos/Pika"
    
    # Run degradation
    degrade_videos(INPUT_DIR, OUTPUT_DIR, downscale_prob=0.8)