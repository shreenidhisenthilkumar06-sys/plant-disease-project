
import os
import shutil
import random
import argparse
from pathlib import Path


CORN_CLASS_MAP = {
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot'    : 'Gray_leaf_spot',
    'Corn_(maize)___Common_rust_'                           : 'Common_rust',
    'Corn_(maize)___healthy'                                : 'Healthy',
    'Corn_(maize)___Northern_Leaf_Blight'                   : 'Leaf_blight',
}

TRAIN_SAMPLES_PER_CLASS = 30 
RANDOM_SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare PlantVillage corn Disease Dataset')
    parser.add_argument('--source_dir', type=str, required=True,
                        help='Path to raw PlantVillage folder (contains corn___* subdirs)')
    parser.add_argument('--output_dir', type=str, default='./data',
                        help='Output directory for train/val/test splits')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED)
    return parser.parse_args()


def split_and_copy(source_dir, output_dir, seed=RANDOM_SEED):
    
    random.seed(seed)

    splits = ['train', 'val', 'test']
    for split in splits:
        for cls in CORN_CLASS_MAP.values():
            os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True)

    summary = {}

    for source_name, class_name in CORN_CLASS_MAP.items():
        src_folder = os.path.join(source_dir, source_name)

        # Try alternate folder structures
        if not os.path.exists(src_folder):
            # Some downloads have a 'color' subfolder
            src_folder = os.path.join(source_dir, 'color', source_name)
        if not os.path.exists(src_folder):
            print(f"  WARNING: Could not find {source_name}, skipping.")
            continue

        images = [f for f in os.listdir(src_folder)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(images)

    
        train_imgs = images[:TRAIN_SAMPLES_PER_CLASS]
        remaining  = images[TRAIN_SAMPLES_PER_CLASS:]
        mid        = len(remaining) // 2
        val_imgs   = remaining[:mid]
        test_imgs  = remaining[mid:]

        for img in train_imgs:
            shutil.copy(os.path.join(src_folder, img),
                        os.path.join(output_dir, 'train', class_name, img))
        for img in val_imgs:
            shutil.copy(os.path.join(src_folder, img),
                        os.path.join(output_dir, 'val', class_name, img))
        for img in test_imgs:
            shutil.copy(os.path.join(src_folder, img),
                        os.path.join(output_dir, 'test', class_name, img))

        summary[class_name] = {
            'train': len(train_imgs),
            'val'  : len(val_imgs),
            'test' : len(test_imgs),
            'total': len(images)
        }
        print(f"  {class_name:25s}: Train={len(train_imgs):4d} | Val={len(val_imgs):4d} | Test={len(test_imgs):4d}")

    print(f"\nDataset prepared at: {output_dir}/")
    print("Structure:")
    print("  data_corn/train/<class>/*.jpg")
    print("  data_corn/val/<class>/*.jpg")
    print("  data_corn/test/<class>/*.jpg")
    return summary


def verify_dataset(data_dir):
    """Print dataset statistics to verify split."""
    print("\n=== Dataset Verification ===")
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            continue
        total = 0
        print(f"\n{split.upper()}:")
        for cls in sorted(os.listdir(split_dir)):
            cls_dir = os.path.join(split_dir, cls)
            if os.path.isdir(cls_dir):
                count = len(os.listdir(cls_dir))
                total += count
                print(f"  {cls:25s}: {count}")
        print(f"  {'TOTAL':25s}: {total}")


if __name__ == '__main__':
    args = parse_args()
    print(f"Preparing dataset from: {args.source_dir}")
    print(f"Output directory      : {args.output_dir}\n")
    split_and_copy(args.source_dir, args.output_dir, args.seed)
    verify_dataset(args.output_dir)
