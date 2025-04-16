"""
Example script showing how to use the traffic density detection system programmatically.
"""
import os
import cv2
from ultralytics import YOLO
from utils import (
    load_image, 
    preprocess_image, 
    calculate_density, 
    visualize_results, 
    save_results
)

# Path to sample traffic image
IMAGE_PATH = "samples/traffic.jpg"
OUTPUT_DIR = "results"
MODEL_PATH = "yolov8n.pt"

def main():
    # Check if image exists
    if not os.path.exists(IMAGE_PATH):
        print(f"Image not found at {IMAGE_PATH}")
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
    output_filename = os.path.basename(IMAGE_PATH).split('.')[0] + "_example_result.jpg"
    output_path = save_results(result_image, OUTPUT_DIR, output_filename)
    print(f"Results saved to: {output_path}")

if __name__ == "__main__":
    main() 