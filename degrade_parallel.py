import subprocess
import random
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from time import perf_counter
import os

# def process_video_fixed(video_file, output_dir, crf_range, downscale_prob):
#     output_file = Path(output_dir) / f"{video_file.stem}.mp4"
    
#     if output_file.exists():
#         return video_file.name, 0, False, True, "Exists"
    
#     crf = random.randint(*crf_range)
#     downscale = random.random() < downscale_prob
    
#     # 简化命令，让FFmpeg自动处理格式
#     cmd = [
#         'ffmpeg', '-y', '-i', str(video_file),
#         '-c:v', 'libx264', '-crf', str(crf),
#         '-pix_fmt', 'yuv420p',
#         '-c:a', 'aac', '-b:a', '128k',
#         '-movflags', '+faststart',
#         str(output_file)
#     ]
    
#     if downscale:
#         cmd.insert(-1, '-vf')
#         cmd.insert(-1, 'scale=iw/2:ih/2')
    
#     subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
#     if output_file.exists():
#         return video_file.name, crf, downscale, True, "Success"
#     return video_file.name, crf, downscale, False, "Failed"

def process_video_fixed(video_file, output_dir, crf_range, downscale_prob):
    output_file = Path(output_dir) / f"{video_file.stem}.mp4"
    # 使用临时文件名，防止写入一半的文件被误认为已完成
    temp_file = Path(output_dir) / f"{video_file.stem}_temp_{random.randint(1000,9999)}.mp4"
    
    # 检查目标文件是否已存在且非空（简单的跳过检查）
    if output_file.exists() and output_file.stat().st_size > 0:
        return video_file.name, 0, False, True, "Exists"
    
    crf = random.randint(*crf_range)
    downscale = random.random() < downscale_prob
    
    cmd = [
        'ffmpeg', '-y', '-i', str(video_file),
        '-c:v', 'libx264', '-crf', str(crf),
        '-pix_fmt', 'yuv420p',
        '-c:a', 'aac', '-b:a', '128k',
        '-movflags', '+faststart',
        str(temp_file) # 先输出到临时文件
    ]
    
    if downscale:
        cmd.insert(-1, '-vf')
        cmd.insert(-1, 'scale=trunc(iw/4)*2:trunc(ih/4)*2')
        # cmd.insert(-1, 'scale=iw/2:ih/2')
    
    try:
        # 捕获返回码，确保 ffmpeg 正常退出 (returncode == 0)
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        if result.returncode == 0 and temp_file.exists():
            # 只有成功了才重命名
            temp_file.replace(output_file)
            return video_file.name, crf, downscale, True, "Success"
        else:
            # 如果失败，清理可能残留的临时文件
            if temp_file.exists():
                os.remove(temp_file)
            return video_file.name, crf, downscale, False, f"FFmpeg Error Code: {result.returncode}"
            
    except Exception as e:
        if temp_file.exists():
            os.remove(temp_file)
        return video_file.name, crf, downscale, False, str(e)

def degrade_videos_parallel_fixed(input_dir, output_dir, crf_range=(28, 50), 
                                downscale_prob=0.5, max_workers=None):
    """Fixed parallel video degradation."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Collect files
    video_files = [f for f in input_path.iterdir() 
                   if f.suffix.lower() in {'.mp4', '.webm'}]
    
    # Count by type
    mp4_count = sum(1 for f in video_files if f.suffix.lower() == '.mp4')
    webm_count = sum(1 for f in video_files if f.suffix.lower() == '.webm')
    
    print(f"Found {len(video_files)} videos: {mp4_count} MP4, {webm_count} WebM")
    print(f"Processing with {max_workers or multiprocessing.cpu_count()} workers")
    print("-" * 60)
    
    # Process
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_video_fixed, f, output_dir, 
                                   crf_range, downscale_prob): f.name 
                  for f in video_files}
        
        stats = {'success': 0, 'failed': 0, 'skipped': 0}
        
        for i, future in enumerate(as_completed(futures), 1):
            name, crf, downscale, success, message = future.result()
            
            if success:
                if "Exists" in message:
                    stats['skipped'] += 1
                    symbol = "-"
                else:
                    stats['success'] += 1
                    symbol = "✓"
                status = "downscaled" if downscale else "normal"
                print(f"[{i}/{len(video_files)}] {symbol} {name}: CRF={crf}, {status}")
            else:
                stats['failed'] += 1
                print(f"[{i}/{len(video_files)}] ✗ {name}: {message}")
    
    # Summary
    print("-" * 60)
    print(f"RESULTS")
    print(f"Success: {stats['success']}")
    print(f"Skipped: {stats['skipped']}")
    print(f"Failed: {stats['failed']}")
    print(f"Output: {output_dir}")

if __name__ == "__main__":
    random.seed(42)
    # INPUT_DIR = "../Data/data_for_joint_sampling/InternVid-AES"
    # OUTPUT_DIR = "../Data/degraded_videos/InternVid-AES"
    INPUT_DIR = "/data1/Data_AIGVDetect/data_for_joint_sampling/Pika_val"
    OUTPUT_DIR = "../Data/degraded_videos/Pika"
    
    start = perf_counter()
    
    # Limit workers to avoid overwhelming system
    degrade_videos_parallel_fixed(
        INPUT_DIR,
        OUTPUT_DIR,
        crf_range=(28, 50),
        downscale_prob=0.5,
        max_workers=64  # Conservative limit
    )
    
    elapsed = perf_counter() - start
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f}min)")