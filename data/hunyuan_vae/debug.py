from data.hunyuan_vae import load_vae


if __name__ == "__main__":
    vae_demo = load_vae(
        vae_type="884-16c-hy",
        vae_precision="fp16",
        device='cuda'
    )
    frame_paths = [f'../Data/GenVideo/video_frames/real/Kinetics-400/train/__mANAQ3Vts_000000_000010/frame{str(i).zfill(4)}.jpg' for i in range(1,33)]
    recon_imgs = vae_demo.reconstruct(frame_paths)
    for idx, img in enumerate(recon_imgs):
        img.save(f'results/vae_reconstruct/hunyuan/recon_frame_{idx+1}.jpg')