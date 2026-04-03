import os
import argparse
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from models.dct import DCT_base_Rec_Module


def main():
    OUTPUT_DIR = f'../Data/RealDist/dct_features'
    # Simple argument parser
    parser = argparse.ArgumentParser(description="DCT feature extraction for images")
    parser.add_argument("txt_path", type=str, help="Path to txt file containing image paths (one per line)")
    args = parser.parse_args()
    
    # Initialize DCT module
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dct = DCT_base_Rec_Module().to(device)
    dct.eval()
    
    # Preprocessing: convert to Tensor only, no resizing (since input sizes vary)
    to_tensor = transforms.ToTensor()
    
    # Read image paths from txt file
    with open(args.txt_path, 'r') as f:
        image_paths = [line.strip() for line in f if line.strip()]
    
    print(f"Found {len(image_paths)} images to process")
    print(f"Output directory: {OUTPUT_DIR}")

    if image_paths:
        first_img_path = image_paths[0]
        first_save_path = first_img_path.replace('.jpg', '.pt').replace('video_frames', 'dct_features')
        if os.path.exists(first_save_path):
            print(f"First feature already exists: {first_save_path}")
            print("Exiting...")
            return
    
    # Process images one by one (not batch processing)
    for img_path in tqdm(image_paths, desc="Processing images"):
        # Check if image exists
        if not os.path.exists(img_path):
            print(f"Warning: Image not found: {img_path}")
            continue
        
        try:
            # Load image
            img = Image.open(img_path).convert('RGB')
            
            # Convert to Tensor [C, H, W], keeping original size
            img_tensor = to_tensor(img).to(device)
            
            # Extract DCT features
            with torch.no_grad():
                x_minmin, x_maxmax, x_minmin1, x_maxmax1 = dct(img_tensor)
            
            # Generate output filename (using original image name)
            save_path = img_path.replace('.jpg', '.pt').replace('video_frames', 'dct_features')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Save features
            features = {
                'x_minmin': x_minmin.cpu(),
                'x_maxmax': x_maxmax.cpu(),
                'x_minmin1': x_minmin1.cpu(),
                'x_maxmax1': x_maxmax1.cpu(),
            }
            torch.save(features, save_path)
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    print("Processing completed!")


if __name__ == "__main__":
    main()
