from loguru import logger
from moviepy import VideoFileClip

import numpy as np
import os
import multiprocessing


def get_video_frame_count(file_path):
    """get total frame count of video"""
    video = VideoFileClip(file_path)
    # print(video.fps, video.duration)
    return int(video.fps * video.duration)

# region uniform sampling
def process_video(args):
    """process single video"""
    video_path, num_frames, output_dir = args  # index, video_path
    if video_path.endswith(".mp4"):
        try:
            output_dir = output_dir 
            os.makedirs(output_dir, exist_ok=True)

            total_frames = get_video_frame_count(video_path)
            frame_interval = max(1, total_frames // num_frames)

            cmd = (
                f"ffmpeg -i {video_path} "
                f"-vf select='not(mod(n\,{frame_interval}))',setpts=N/FRAME_RATE/TB "
                f"-vframes {num_frames} {output_dir}/frame%d.jpg"
                # f" > /dev/null 2>&1"
            )
            ret = os.system(cmd)
        except Exception as e:
            print(f"Error processing {video_path}: {str(e)}")
    elif video_path.endswith(".gif"):
        try:
            output_dir = os.path.join(output_dir)
            os.makedirs(output_dir, exist_ok=True)

            total_frames = get_video_frame_count(video_path)
            frame_interval = max(1, total_frames // num_frames)

            cmd = (
                f"ffmpeg -i {video_path} "
                f"-vf select='not(mod(n\,{frame_interval}))',setpts=N/FRAME_RATE/TB "
                f"-vframes {num_frames} {output_dir}/frame%d.jpg"
                f" > /dev/null 2>&1"
            )
            os.system(cmd)
        except Exception as e:
            print(f"Error processing {video_path}: {str(e)}")

def process_video2frames(dir, ids, num_frames, output_base_dir):
    video_args = [(os.path.join(dir, f"{id}.mp4"), num_frames, os.path.join(output_base_dir, id)) for id in ids]
    logger.info(f"Processing {len(ids)} videos")
    pool = multiprocessing.Pool(processes=24)
    pool.map(process_video, video_args)
    pool.close()
    pool.join()

def setup_dataset(data_path='../Data/myvideos', generation_model='MSR-VTT', label='real', mode='test', len_load=None, num_frames=8):
    assert os.path.exists(data_path), f"Data path {data_path} does not exist"
    video_ids_txt = os.path.join(data_path, "split", label, generation_model, f"{mode}_ids.txt")
    assert os.path.exists(video_ids_txt), f"video_ids_txt not found"
    with open(video_ids_txt, "r") as f:
        video_ids = [line.strip() for line in f if line.strip()]
    video_ids = [vid for vid in video_ids if vid.endswith('.mp4')]
    if len_load is not None:
        video_ids = video_ids[:len_load]
    video_ids = [os.path.splitext(video_id)[0] for video_id in video_ids]
    logger.info(f"len of video_ids is {len(video_ids)}")
    video_dir = os.path.join(data_path, "video", label, generation_model)
    frame_dir = os.path.join(data_path, "video_frames", label, generation_model, mode)
    unproceesed_ids = [
                video_id for video_id in video_ids
                if not os.path.isdir(os.path.join(frame_dir, video_id)) or len(os.listdir(os.path.join(frame_dir, video_id))) != num_frames
    ]
    if len(unproceesed_ids) > 0:
        process_video2frames(video_dir, unproceesed_ids, num_frames, frame_dir)
# endregion

# region consecutive sampling
def process_video_consecutive(args):
    """process single video with consecutive frames from start"""
    video_path, num_frames, output_dir = args
    if video_path.endswith((".mp4", ".gif")):
        try:
            os.makedirs(output_dir, exist_ok=True)
            cmd = (
                f"ffmpeg -i {video_path} "
                f"-vframes {num_frames} "
                f"-q:v 2 "
                f"{output_dir}/frame%d.jpg"
                f" > /dev/null 2>&1"
            )
            ret = os.system(cmd)
            if ret != 0:
                print(f"FFmpeg error processing {video_path}")
        except Exception as e:
            print(f"Error processing {video_path}: {str(e)}")

def process_video2frames_consecutive(dir, ids, num_frames, output_base_dir):
    video_args = [(os.path.join(dir, f"{id}.mp4"), num_frames, os.path.join(output_base_dir, id)) for id in ids]
    logger.info(f"Processing {len(ids)} videos")
    pool = multiprocessing.Pool(processes=24)
    pool.map(process_video_consecutive, video_args)
    pool.close()
    pool.join()

def setup_dataset_consecutive(data_path='../Data/myvideos', generation_model='MSR-VTT', label='real', mode='test', len_load=None, num_frames=8):
    assert os.path.exists(data_path), f"Data path {data_path} does not exist"
    video_ids_txt = os.path.join(data_path, "split", label, generation_model, f"{mode}_ids.txt")
    assert os.path.exists(video_ids_txt), f"video_ids_txt not found"
    with open(video_ids_txt, "r") as f:
        video_ids = [line.strip() for line in f if line.strip()]
    video_ids = [vid for vid in video_ids if vid.endswith('.mp4')]
    if len_load is not None:
        video_ids = video_ids[:len_load]
    video_ids = [os.path.splitext(video_id)[0] for video_id in video_ids]
    logger.info(f"len of video_ids is {len(video_ids)}")
    video_dir = os.path.join(data_path, "video", label, generation_model)
    frame_dir = os.path.join(data_path, "consecutive_frames", label, generation_model, mode)
    unproceesed_ids = [
                video_id for video_id in video_ids
                if not os.path.isdir(os.path.join(frame_dir, video_id)) or len(os.listdir(os.path.join(frame_dir, video_id))) != num_frames
    ]
    if len(unproceesed_ids) > 0:
        process_video2frames_consecutive(video_dir, unproceesed_ids, num_frames, frame_dir)
# endregion

# region extract all frames
def extract_all_frames(args):
    """Extract all frames from the entire video"""
    video_path, output_dir = args
    supported_formats = (".mp4", ".gif", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv")
    
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
        for ext in ['.mp4', '.mov']:
            potential_path = os.path.join(dir, f"{video_id}{ext}")
            if os.path.exists(potential_path):
                video_path = potential_path
                break
        
        if video_path and os.path.exists(video_path):
            video_args.append((video_path, os.path.join(output_base_dir, video_id)))
        else:
            logger.warning(f"Video file not found for ID: {video_id}, tried extensions: .mp4, .mov")
    
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
    assert os.path.exists(video_ids_txt), f"video_ids_txt not found"
    with open(video_ids_txt, "r") as f:
        video_ids = [line.strip() for line in f if line.strip()]

    video_ids = [vid for vid in video_ids if vid.endswith(('.mp4', '.mov'))]
    
    video_ids = [os.path.splitext(video_id)[0] for video_id in video_ids]
    logger.info(f"len of video_ids is {len(video_ids)}")
    video_dir = os.path.join(data_path, "video", label, generation_model)
    frame_dir = os.path.join(data_path, "video_frames", label, generation_model, mode)
    unproceesed_ids = [
        video_id for video_id in video_ids
        if not os.path.isdir(os.path.join(frame_dir, video_id))
    ]
    if len(unproceesed_ids) > 0:
        process_video_in_parallel(video_dir, unproceesed_ids, frame_dir)
# endregion


if __name__ == "__main__":
    logger.debug("This module is not meant to be run directly. Import it in your code to use the functions and classes defined here.")
    # video_path, output_dir = '../Data/demo2extract/ZYc410CE4Rg_000001_000011.mp4', '../Data/extracted_frames'
    # extract_all_frames((video_path, output_dir))
    dataset_frame_extract(data_path='../Data/myvideos', generation_model='MSR-VTT', label='real', mode='test')