import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import shutil

def add_salt_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):
    """Add salt and pepper noise to an image"""
    noisy = np.copy(image)
    
    # Salt noise (white pixels)
    salt_mask = np.random.random(image.shape[:2]) < salt_prob
    noisy[salt_mask] = 255
    
    # Pepper noise (black pixels)
    pepper_mask = np.random.random(image.shape[:2]) < pepper_prob
    noisy[pepper_mask] = 0
    
    return noisy

def add_blur(image, kernel_size=5):
    """Add Gaussian blur to an image"""
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def add_motion_blur(image, kernel_size=5):
    """Add motion blur to an image"""
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    kernel = kernel / kernel_size
    return cv2.filter2D(image, -1, kernel)

def convert_csv_to_yolo_format(csv_file, image_dir, output_label_dir, image_width=676, image_height=380):
    """Convert CSV annotations to YOLO format (normalized)"""
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_label_dir, exist_ok=True)
    
    # Group by image name
    grouped = df.groupby('image')
    
    for image_name, group in tqdm(grouped, desc="Converting annotations"):
        # Get all bounding boxes for this image
        bboxes = group[['xmin', 'ymin', 'xmax', 'ymax']].values
        
        # Convert to YOLO format (class_id, x_center, y_center, width, height) - normalized
        yolo_annotations = []
        for box in bboxes:
            xmin, ymin, xmax, ymax = box
            
            # Ensure coordinates are within image bounds
            xmin = max(0, min(xmin, image_width))
            xmax = max(0, min(xmax, image_width))
            ymin = max(0, min(ymin, image_height))
            ymax = max(0, min(ymax, image_height))
            
            # Calculate normalized center coordinates and dimensions
            x_center = ((xmin + xmax) / 2) / image_width
            y_center = ((ymin + ymax) / 2) / image_height
            width = (xmax - xmin) / image_width
            height = (ymax - ymin) / image_height
            
            # Class ID 0 for vehicle
            yolo_annotations.append(f"0 {x_center} {y_center} {width} {height}")
        
        # Write annotations to a text file
        output_file = os.path.join(output_label_dir, os.path.splitext(image_name)[0] + '.txt')
        with open(output_file, 'w') as f:
            f.write('\n'.join(yolo_annotations))

def create_augmented_dataset(input_image_dir, input_label_dir, output_dir, augmentation_factor=3):
    """Create an augmented dataset with blurry, noisy, and motion-blurred images"""
    # Create output directories
    output_image_dir = os.path.join(output_dir, 'images')
    output_label_dir = os.path.join(output_dir, 'labels')
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)
    
    # Process images
    image_files = [f for f in os.listdir(input_image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    for img_file in tqdm(image_files, desc="Augmenting images"):
        # Load image
        img_path = os.path.join(input_image_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            continue
        
        # Get base filename without extension
        base_name = os.path.splitext(img_file)[0]
        
        # Copy original image
        cv2.imwrite(os.path.join(output_image_dir, img_file), img)
        
        # Copy original label if exists
        label_file = base_name + '.txt'
        label_path = os.path.join(input_label_dir, label_file)
        if os.path.exists(label_path):
            shutil.copy(label_path, os.path.join(output_label_dir, label_file))
        
        # Generate augmented versions
        for i in range(augmentation_factor):
            # Create different augmentation types
            
            # 1. Salt and pepper noise only
            if i % augmentation_factor == 0:
                salt_prob = np.random.uniform(0.01, 0.03)
                pepper_prob = np.random.uniform(0.01, 0.03)
                noisy_img = add_salt_pepper_noise(img, salt_prob, pepper_prob)
                aug_name = f"{base_name}_noise{i}.jpg"
                cv2.imwrite(os.path.join(output_image_dir, aug_name), noisy_img)
            
            # 2. Gaussian blur only
            elif i % augmentation_factor == 1:
                blur_level = np.random.choice([3, 5, 7, 9])
                blurry_img = add_blur(img, blur_level)
                aug_name = f"{base_name}_blur{i}.jpg"
                cv2.imwrite(os.path.join(output_image_dir, aug_name), blurry_img)
            
            # 3. Motion blur
            else:
                motion_blur_level = np.random.choice([3, 5, 7, 9])
                motion_blurred_img = add_motion_blur(img, motion_blur_level)
                aug_name = f"{base_name}_motionblur{i}.jpg"
                cv2.imwrite(os.path.join(output_image_dir, aug_name), motion_blurred_img)
            
            # Copy label file for the augmented image (same annotations)
            if os.path.exists(label_path):
                aug_label = os.path.splitext(aug_name)[0] + '.txt'
                shutil.copy(label_path, os.path.join(output_label_dir, aug_label))

def split_dataset(dataset_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """Split dataset into train, validation and test sets"""
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1"
    
    # Create output directories
    train_dir = os.path.join(dataset_dir, 'train')
    val_dir = os.path.join(dataset_dir, 'val')
    test_dir = os.path.join(dataset_dir, 'test')
    
    for split_dir in [train_dir, val_dir, test_dir]:
        os.makedirs(os.path.join(split_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(split_dir, 'labels'), exist_ok=True)
    
    # Get all image files
    images_dir = os.path.join(dataset_dir, 'images')
    labels_dir = os.path.join(dataset_dir, 'labels')
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Shuffle files
    np.random.shuffle(image_files)
    
    # Calculate split indices
    n_total = len(image_files)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)
    
    # Split files
    train_files = image_files[:n_train]
    val_files = image_files[n_train:n_train+n_val]
    test_files = image_files[n_train+n_val:]
    
    # Copy files to respective directories
    for files, target_dir in [(train_files, train_dir), 
                             (val_files, val_dir), 
                             (test_files, test_dir)]:
        for img_file in tqdm(files, desc=f"Copying to {os.path.basename(target_dir)}"):
            # Copy image
            shutil.copy(
                os.path.join(images_dir, img_file),
                os.path.join(target_dir, 'images', img_file)
            )
            
            # Copy corresponding label if exists
            label_file = os.path.splitext(img_file)[0] + '.txt'
            label_path = os.path.join(labels_dir, label_file)
            if os.path.exists(label_path):
                shutil.copy(
                    label_path,
                    os.path.join(target_dir, 'labels', label_file)
                )

def main():
    # Input directories
    input_csv = "/Users/aryanshah/Downloads/AI-PROJECT-master/AI-PROJECT/data/raw/train_solution_bounding_boxes.csv"
    input_image_dir = "/Users/aryanshah/Downloads/AI-PROJECT-master/AI-PROJECT/data/raw/training_images"
    
    # Output directories
    output_label_dir = "data/yolo_labels"
    augmented_dataset_dir = "data/augmented_dataset"
    
    # Step 1: Convert CSV annotations to YOLO format
    print("Converting CSV annotations to YOLO format...")
    convert_csv_to_yolo_format(input_csv, input_image_dir, output_label_dir)
    
    # Step 2: Create augmented dataset with blurry, noisy, and motion-blurred images
    print("Creating augmented dataset...")
    create_augmented_dataset(input_image_dir, output_label_dir, augmented_dataset_dir, augmentation_factor=3)
    
    # Step 3: Split dataset into train, val, test
    print("Splitting dataset...")
    split_dataset(augmented_dataset_dir)
    
    print("Data preprocessing complete!")

if __name__ == "__main__":
    main()