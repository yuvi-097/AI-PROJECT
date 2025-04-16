import os
import cv2
import numpy as np
import argparse
from ultralytics import YOLO
import matplotlib.pyplot as plt
from data_preprocessing import add_salt_pepper_noise, add_blur

def parse_args():
    parser = argparse.ArgumentParser(description="Test YOLOv8 model on blurry images with salt and pepper noise")
    parser.add_argument('--model', type=str, required=True, help='Path to trained model weights')
    parser.add_argument('--image', type=str, required=True, help='Path to test image')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    parser.add_argument('--blur-level', type=int, default=9, help='Blur kernel size (odd number)')
    parser.add_argument('--salt-prob', type=float, default=0.02, help='Salt noise probability')
    parser.add_argument('--pepper-prob', type=float, default=0.02, help='Pepper noise probability')
    
    return parser.parse_args()

def create_test_variations(image_path, output_dir, blur_level=9, salt_prob=0.02, pepper_prob=0.02):
    """Create variations of the test image: original, blurry, noisy, and blurry+noisy"""
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Original image
    original_path = os.path.join(output_dir, 'original.jpg')
    cv2.imwrite(original_path, img)
    
    # Blurry image
    blurry_img = add_blur(img, blur_level)
    blurry_path = os.path.join(output_dir, 'blurry.jpg')
    cv2.imwrite(blurry_path, blurry_img)
    
    # Noisy image
    noisy_img = add_salt_pepper_noise(img, salt_prob, pepper_prob)
    noisy_path = os.path.join(output_dir, 'noisy.jpg')
    cv2.imwrite(noisy_path, noisy_img)
    
    # Blurry + noisy image
    blurry_noisy_img = add_blur(add_salt_pepper_noise(img, salt_prob, pepper_prob), blur_level)
    blurry_noisy_path = os.path.join(output_dir, 'blurry_noisy.jpg')
    cv2.imwrite(blurry_noisy_path, blurry_noisy_img)
    
    return {
        'original': {'image': img, 'path': original_path},
        'blurry': {'image': blurry_img, 'path': blurry_path},
        'noisy': {'image': noisy_img, 'path': noisy_path},
        'blurry_noisy': {'image': blurry_noisy_img, 'path': blurry_noisy_path}
    }

def run_detection(model, test_images, conf_threshold, output_dir):
    """Run object detection on each test image variation"""
    # Results dictionary
    results = {}
    
    for name, img_data in test_images.items():
        print(f"Running detection on {name} image...")
        # Run prediction
        prediction = model(img_data['path'], conf=conf_threshold, save=True, project=output_dir, name=name)
        
        # Get detection counts
        detection_count = len(prediction[0].boxes)
        
        # Store results
        results[name] = {
            'prediction': prediction[0],
            'detection_count': detection_count
        }
        
        print(f"  Detected {detection_count} vehicles")
    
    return results

def compare_results(results, output_dir):
    """Compare and visualize detection results"""
    # Create summary image
    fig, axs = plt.subplots(2, 2, figsize=(20, 15))
    axs = axs.flatten()
    
    for i, (name, result) in enumerate(results.items()):
        # Get prediction image with boxes
        pred_image = result['prediction'].plot()
        pred_image = cv2.cvtColor(pred_image, cv2.COLOR_BGR2RGB)
        
        # Add to plot
        axs[i].imshow(pred_image)
        axs[i].set_title(f"{name.capitalize()}: {result['detection_count']} vehicles detected")
        axs[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison.png'))
    plt.close()
    
    # Create CSV summary
    with open(os.path.join(output_dir, 'detection_summary.txt'), 'w') as f:
        f.write("Image Type,Vehicles Detected\n")
        for name, result in results.items():
            f.write(f"{name},{result['detection_count']}\n")
    
    print(f"Comparison saved to {os.path.join(output_dir, 'comparison.png')}")
    print(f"Summary saved to {os.path.join(output_dir, 'detection_summary.txt')}")

def main():
    # Parse arguments
    args = parse_args()
    
    # Load model
    print(f"Loading model from {args.model}...")
    model = YOLO(args.model)
    
    # Create test image variations
    print(f"Creating test variations of {args.image}...")
    test_images = create_test_variations(
        args.image, 
        args.output, 
        args.blur_level, 
        args.salt_prob, 
        args.pepper_prob
    )
    
    # Run detection on each variation
    detection_results = run_detection(model, test_images, args.conf, args.output)
    
    # Compare and visualize results
    compare_results(detection_results, args.output)
    
    print("\nTesting complete!")

if __name__ == "__main__":
    main() 