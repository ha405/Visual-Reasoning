# File: preprocess_data.py
import os
from PIL import Image
from tqdm import tqdm
import sys

# ==============================================================================
# 1. CONFIGURE THESE THREE VARIABLES
# ==============================================================================
# Set the path to your original, downloaded Office-Home dataset folder
SOURCE_DIR = r"D:\Haseeb\Datasets\OfficeHomeDataset_10072016" 

# Set the path and name for the NEW folder where processed images will be saved
TARGET_DIR = r"C:\Users\Haseeb\OneDrive - Higher Education Commission\sproj\Visual-Reasoning\pre-processed_datasets\office_home"

# Set the desired image size
IMG_SIZE = (224, 224)
# ==============================================================================

def preprocess_dataset(source_root, target_root, size):
    if not os.path.isdir(source_root):
        print(f"FATAL ERROR: Source directory not found at '{source_root}'")
        print("Please update the SOURCE_DIR variable in this script.")
        sys.exit(1)

    os.makedirs(target_root, exist_ok=True)
    print(f"Target directory created/found at: {target_root}")

    domains = [d for d in os.listdir(source_root) if os.path.isdir(os.path.join(source_root, d))]
    total_files = 0
    corrupt_files = 0

    print(f"Starting pre-processing for {len(domains)} domains...")

    for domain in domains:
        source_domain_path = os.path.join(source_root, domain)
        target_domain_path = os.path.join(target_root, domain)
        print(f"\nProcessing domain: {domain}")
        
        classes = [c for c in os.listdir(source_domain_path) if os.path.isdir(os.path.join(source_domain_path, c))]
        for class_name in classes:
            source_class_path = os.path.join(source_domain_path, class_name)
            target_class_path = os.path.join(target_domain_path, class_name)
            os.makedirs(target_class_path, exist_ok=True)
            
            image_files = os.listdir(source_class_path)
            for img_name in tqdm(image_files, desc=f"  - Class: {class_name}", unit="img"):
                source_img_path = os.path.join(source_class_path, img_name)
                target_img_path = os.path.join(target_class_path, img_name)
                total_files += 1

                try:
                    with Image.open(source_img_path) as img:
                        img_rgb = img.convert('RGB')
                        img_resized = img_rgb.resize(size, Image.Resampling.LANCZOS)
                        img_resized.save(target_img_path)
                except Exception as e:
                    corrupt_files += 1
                    print(f"\nWARNING: Skipping corrupt file: {source_img_path} | Error: {e}")

    print("\n" + "="*50)
    print("Pre-processing complete!")
    print(f"Total files checked: {total_files}")
    print(f"Corrupt files found and skipped: {corrupt_files}")
    print(f"Clean, resized dataset saved at: {target_root}")
    print("="*50)

if __name__ == '__main__':
    preprocess_dataset(SOURCE_DIR, TARGET_DIR, IMG_SIZE)