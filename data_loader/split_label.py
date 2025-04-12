import os
import random
from collections import defaultdict
import argparse
import shutil

def get_domain_dict(dataset_type):

    if dataset_type.lower() == 'officehome':
        return {
            'Product': 'p',
            'Art': 'a',
            'Clipart': 'c',
            'Real_World': 'r',
        }
    elif dataset_type.lower() == 'vlcs':
        return {
            'VOC2007': 'v',
            'LabelMe': 'l',
            'Caltech101': 'c',
            'SUN09': 's',
        }
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}. Choose from'officehome', or 'vlcs'.")

def generate_labels(root_path, dataset_type):

    # Get domain mapping based on dataset type
    domain_dict = get_domain_dict(dataset_type)

    print(f"Processing {dataset_type.upper()} dataset with domains: {', '.join(domain_dict.keys())}")

    # Create splits directory if it doesn't exist
    splits_dir = os.path.join(root_path, 'splits')
    os.makedirs(splits_dir, exist_ok=True)

    # Create images directory if it doesn't exist
    images_dir = os.path.join(root_path, 'images')
    os.makedirs(images_dir, exist_ok=True)

    # Process each domain
    for domain_name, domain_code in domain_dict.items():
        domain_path = os.path.join(root_path, domain_name)
        if not os.path.isdir(domain_path):
            print(f"Warning: Domain directory {domain_path} not found. Skipping.")
            continue

        print(f"Processing domain: {domain_name}")

        # Dictionary to store images by class
        class_images = defaultdict(list)

        # Find all class directories
        class_dirs = [d for d in os.listdir(domain_path) if os.path.isdir(os.path.join(domain_path, d))]
        class_dirs.sort()  # Ensure consistent class ordering

        # Create class to label mapping
        class_to_label = {cls: idx for idx, cls in enumerate(class_dirs)}

        # Find all images in each class
        for class_name in class_dirs:
            class_path = os.path.join(domain_path, class_name)

            # Find all images in this class directory
            for root, _, files in os.walk(class_path):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        # Get relative path for image
                        img_path = os.path.join(domain_name, class_name, file)
                        class_images[class_name].append(img_path)

        # Create a list of all images with their labels
        all_images = []
        for class_name, images in class_images.items():
            label = class_to_label[class_name]
            for img_path in images:
                all_images.append((img_path, label))

        # Shuffle the images
        random.seed(42)  # For reproducibility
        random.shuffle(all_images)

        # Split into train (70%), val (10%), test (20%)
        train_split = int(len(all_images) * 0.7)
        val_split = train_split + int(len(all_images) * 0.1)


        train_images = all_images[:train_split]
        val_images = all_images[:train_split]
        test_images = all_images

        # Write split files
        splits = {
            'train': train_images,
            'val': val_images,
            'test': test_images
        }

        for split_name, images in splits.items():
            split_file = os.path.join(splits_dir, f"{domain_name}_{split_name}.txt")
            with open(split_file, 'w') as f:
                for img_path, label in images:
                    f.write(f"{img_path} {label}\n")

            print(f"Created {split_file} with {len(images)} images")

    print("Label generation complete!")
    print(f"Split files are located in: {splits_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate label files for dataset')
    parser.add_argument('--root_path', type=str, help='Root path of dataset')
    parser.add_argument('--dataset', type=str, default='officehome', choices=['officehome', 'vlcs'],
                        help='Dataset type to process (officehome, or vlcs)')
    args = parser.parse_args()

    generate_labels(args.root_path, args.dataset)
