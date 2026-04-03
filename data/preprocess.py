from loguru import logger
from moviepy import VideoFileClip

import numpy as np
import os
import multiprocessing


SUPPORT_EXT = ['.mp4', '.mov', '.webm', '.gif']

def get_video_frame_count(file_path):
    """get total frame count of video"""
    video = VideoFileClip(file_path)
    # print(video.fps, video.duration)
    return int(video.fps * video.duration)

# region frame extract
def extract_all_frames(args):
    """Extract all frames from the entire video"""
    video_path, output_dir = args
    supported_formats = tuple(SUPPORT_EXT)
    
    if video_path.lower().endswith(supported_formats):
        try:
            os.makedirs(output_dir, exist_ok=True)
            # Extract all frames without limiting the number
            cmd = (
                f"ffmpeg -i {video_path} "
                f"-q:v 2 "  # Set image quality
                f"{output_dir}/frame%04d.jpg"
                f" > /dev/null 2>&1"
            )
            ret = os.system(cmd)
            if ret != 0:
                print(f"FFmpeg error processing {video_path}")
        except Exception as e:
            print(f"Error processing {video_path}: {str(e)}")
    else:
        print(f"Unsupported video format: {video_path}. Supported formats: {', '.join(supported_formats)}")

def process_video_in_parallel(dir, ids, output_base_dir):
    video_args = []
    for video_id in ids:
        video_path = None
        for ext in SUPPORT_EXT:
            potential_path = os.path.join(dir, f"{video_id}{ext}")
            if os.path.exists(potential_path):
                video_path = potential_path
                break
        
        if video_path and os.path.exists(video_path):
            video_args.append((video_path, os.path.join(output_base_dir, video_id)))
        else:
            logger.warning(f"Video file not found for ID: {video_id}, tried extensions: {SUPPORT_EXT}")
    
    if video_args:
        logger.info(f"Processing {len(video_args)} videos")
        pool = multiprocessing.Pool(processes=24)
        pool.map(extract_all_frames, video_args)
        pool.close()
        pool.join()
    else:
        logger.warning("No valid video files found to process")

def dataset_frame_extract(data_path='../Data/myvideos', generation_model='MSR-VTT', label='real', mode='test'):
    assert os.path.exists(data_path), f"Data path {data_path} does not exist"
    video_ids_txt = os.path.join(data_path, "split", label, generation_model, f"{mode}_ids.txt")
    assert os.path.exists(video_ids_txt), f"{video_ids_txt} not found"
    with open(video_ids_txt, "r") as f:
        video_ids = [line.strip() for line in f if line.strip()]

    video_ids = [vid for vid in video_ids if vid.endswith(tuple(SUPPORT_EXT))]
    
    video_ids = [os.path.splitext(video_id)[0] for video_id in video_ids]
    logger.info(f"[{generation_model} / {mode} / {len(video_ids)} videos]")
    if len(video_ids) == 0:
        breakpoint()
    video_dir = os.path.join(data_path, "video", label, generation_model)
    frame_dir = os.path.join(data_path, "video_frames", label, generation_model, mode)
    unproceesed_ids = [
        video_id for video_id in video_ids
        if not os.path.isdir(os.path.join(frame_dir, video_id))
    ]
    if len(unproceesed_ids) > 0:
        process_video_in_parallel(video_dir, unproceesed_ids, frame_dir)
    else:
        logger.info("Videos processed already.")
# endregion


if __name__ == "__main__":
    logger.debug("This module is not meant to be run directly. Import it in your code to use the functions and classes defined here.")
    # video_path, output_dir = '../Data/demo2extract/ZYc410CE4Rg_000001_000011.mp4', '../Data/extracted_frames'
    # extract_all_frames((video_path, output_dir))
    dataset_frame_extract(data_path='../Data/myvideos', generation_model='MSR-VTT', label='real', mode='test')