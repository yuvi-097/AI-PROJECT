"""
Example script showing how to use the traffic density detection system programmatically.
"""
import os
import sys
import cv2
import matplotlib.pyplot as plt

# Add parent directory to PATH to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics import YOLO
from utils import (
    load_image, 
    preprocess_image, 
    calculate_density, 
    visualize_results, 
    
    save_results
)

# Path to sample traffic image - replace with your own image
IMAGE_PATH = "samples/traffic.jpg"  # You'll need to add your own image
OUTPUT_DIR = "results"
MODEL_PATH = "yolov8n.pt"  # Will be downloaded if not present

def main():
    # Check if image exists
    if not os.path.exists(IMAGE_PATH):
        print(f"Please add a traffic image at {IMAGE_PATH} or update the IMAGE_PATH variable")
        print("You can download sample traffic images from various datasets like COCO or BDD100K")
        return
    
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # Load YOLO model
    print(f"Loading YOLO model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    
    # Load and preprocess image
    print(f"Processing image: {IMAGE_PATH}")
    original_image = load_image(IMAGE_PATH)
    preprocessed_image, original_image, image_shape = preprocess_image(original_image)
    
    # Detect vehicles
    print("Detecting vehicles...")
    results = model(original_image, conf=0.25)[0]
    
    # Extract detections
    detections = []
    for det in results.boxes.data.cpu().numpy():
        x1, y1, x2, y2, conf, class_id = det
        detections.append([x1, y1, x2, y2, conf, class_id])
    
    # Calculate density
    print("Calculating traffic density...")
    density_metrics = calculate_density(detections, image_shape)
    print(f"Vehicle count: {density_metrics['vehicle_count']}")
    print(f"Density: {density_metrics['density_percentage']:.2f}%")
    print(f"Density class: {density_metrics['density_class']}")
    
    # Visualize results
    print("Visualizing results...")
    result_image = visualize_results(original_image, detections, density_metrics)
    
    # Save results
    output_filename = os.path.basename(IMAGE_PATH).split('.')[0] + "_result.jpg"
    output_path = save_results(result_image, OUTPUT_DIR, output_filename)
    print(f"Results saved to: {output_path}")
    
    # Display result (optional - works in Jupyter notebook or when running with GUI)
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(f"Vehicle Count: {density_metrics['vehicle_count']} | Density: {density_metrics['density_percentage']:.2f}% | Class: {density_metrics['density_class']}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "plot_" + output_filename))
    plt.show()

if __name__ == "__main__":
    main() 