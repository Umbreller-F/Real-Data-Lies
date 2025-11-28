from data.wan2_2_vae import Wan2_2_VAE
from data.hunyuan_vae import load_vae
from multiprocessing import Pool, cpu_count

import os
import torch
import time
import multiprocessing as mp

# Global variable to store VAE instance per process
vae_instance = None

def init_worker(vae_model):
    """Initialize VAE instance once per worker process"""
    global vae_instance
    if vae_instance is None:
        process_id = mp.current_process().pid
        print(f"Initializing VAE for process {process_id}")
        if vae_model == 'Wan2.2':
            vae_instance = Wan2_2_VAE(
                vae_pth=os.path.join('./ckpts', 'Wan2.2_VAE.pth'),
                device=torch.device("cuda")
            )
        elif vae_model == 'Hunyuan':
            vae_instance = load_vae(
                vae_type="884-16c-hy",
                vae_precision="fp16",
                device='cuda'
            )

def process_single_video(args):
    """Process a single video folder"""
    v_f, save_dir = args
    try:
        global vae_instance
        
        if vae_instance is None:
            return f"{v_f}: error - VAE not initialized"
        
        # frame_paths = [os.path.join(v_f, f"frame{i}.jpg") for i in range(1, 9)]
        frame_paths = sorted(
            [os.path.join(v_f, f) for f in os.listdir(v_f) if f.endswith('.jpg')],
            key=lambda x: int(os.path.splitext(os.path.basename(x))[0].replace('frame', ''))
        )
        frame_paths = frame_paths[:32]
        
        # Check if all frame files exist
        if not all(os.path.exists(fp) for fp in frame_paths):
            print(f"Skipping {v_f} - missing frame files")
            return f"{v_f}: skipped (missing frames)"
        
        video_name = os.path.basename(v_f)
        print(f"Processing: {video_name}")

        save_folder = os.path.join(save_dir, video_name)
        if os.path.exists(save_folder):
            return f"{v_f}: success"
        
        recon_imgs = vae_instance.reconstruct(frame_paths)
        
        os.makedirs(save_folder, exist_ok=True)
        for idx, img in enumerate(recon_imgs):
            img.save(os.path.join(save_folder, f'frame{idx+1}.jpg'))
        
        return f"{v_f}: success"
        
    except Exception as e:
        # print(f"{v_f}: error - {str(e)}")
        return f"{v_f}: error - {str(e)}"


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    vae_model = 'Wan2.2'
    real_domain = 'Kinetics-400'
    split = 'val'
    chunk_index = 0
    chunk_size = 3400
    num_processes = 5
    
    real_video_frames_dir = f'../Data/GenVideo/video_frames/real/{real_domain}/{split}'
    save_dir = f'../Data/GenVideo/video_frames/fake/VAE/{vae_model}/{split}'
    
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Get list of video folders
    video_folders = sorted([os.path.join(real_video_frames_dir, d) 
                    for d in os.listdir(real_video_frames_dir)])
    video_folders = video_folders[chunk_index * chunk_size:(chunk_index+1) * chunk_size]
    
    print(f"Found {len(video_folders)} video folders")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {num_processes} processes")
    
    start_time = time.perf_counter()
    
    # Prepare arguments
    args_list = [(v_f, save_dir) for v_f in video_folders]
    
    # Use initializer to set up VAE once per worker
    with Pool(processes=num_processes, initializer=init_worker, initargs=(vae_model,)) as pool:
        results = pool.map(process_single_video, args_list)
    
    # Print results summary
    success_count = sum(1 for r in results if "success" in r)
    skip_count = sum(1 for r in results if "skipped" in r)
    error_count = sum(1 for r in results if "error" in r)
    
    print(f"\nProcessing completed:")
    print(f"Success: {success_count}, Skipped: {skip_count}, Errors: {error_count}")
    
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"Total execution time: {execution_time:.4f} seconds")
    print(f"Average per video: {execution_time/len(video_folders):.4f} seconds")