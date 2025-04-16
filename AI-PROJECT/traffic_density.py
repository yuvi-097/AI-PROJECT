import argparse
import os
import time
import cv2
import numpy as np
from pathlib import Path
import torch
from ultralytics import YOLO
import sys
import json

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.utils import (
    load_image,
    preprocess_image,
    calculate_density,
    visualize_results,
    save_results
)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Traffic Vehicle Density Detection")
    parser.add_argument("--image", required=True, help="Path to the traffic image")
    parser.add_argument("--output", default="results", help="Output directory for results")
    parser.add_argument("--model", default="yolov8n.pt", help="Path to YOLO model")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--device", default="", help="Device to run inference on (cuda device, i.e. 0 or cpu)")
    parser.add_argument("--json_output", action='store_true', help="Output final metrics as JSON to stdout")
    
    return parser.parse_args()

def load_model(model_path, device=""):
    """
    Load YOLO model
    
    Args:
        model_path (str): Path to the YOLO model weights
        device (str): Device to run inference on (cuda device or cpu)
        
    Returns:
        YOLO: Loaded model
    """
    # Download the model if it doesn't exist and it's a standard YOLO model name
    if not os.path.exists(model_path) and model_path.startswith("yolov8"):
        print(f"Downloading {model_path}...")
    
    # Load the model
    model = YOLO(model_path)
    
    # Set device if specified
    if device:
        model.to(device)
        
    return model

def detect_vehicles(model, image, conf_threshold=0.25):
    """
    Detect vehicles in the image using YOLO
    
    Args:
        model (YOLO): Loaded YOLO model
        image (np.ndarray): Input image
        conf_threshold (float): Confidence threshold for detections
        
    Returns:
        list: List of detections [x1, y1, x2, y2, confidence, class_id]
    """
    # Run inference
    results = model(image, conf=conf_threshold)[0]
    
    # Extract detections
    detections = []
    for det in results.boxes.data.cpu().numpy():
        x1, y1, x2, y2, conf, class_id = det
        detections.append([x1, y1, x2, y2, conf, class_id])
    
    return detections

def main():
    # Parse arguments
    args = parse_arguments()
    
    # Set device
    device = args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu")
    if not args.json_output:
        print(f"Using device: {device}")
    
    # Load model
    if not args.json_output:
        print(f"Loading YOLO model: {args.model}")
    model = load_model(args.model, device)
    
    # Load and preprocess image
    if not args.json_output:
        print(f"Processing image: {args.image}")
    original_image = load_image(args.image)
    preprocessed_image, original_image, image_shape = preprocess_image(original_image)
    
    # Detect vehicles
    if not args.json_output:
        print("Detecting vehicles...")
    start_time = time.time()
    detections = detect_vehicles(model, original_image, args.conf)
    inference_time = time.time() - start_time
    if not args.json_output:
        print(f"Detection completed in {inference_time:.2f} seconds")
    
    # Calculate density
    if not args.json_output:
        print("Calculating traffic density...")
    density_metrics = calculate_density(detections, image_shape)
    if not args.json_output:
        print(f"Vehicle count: {density_metrics['vehicle_count']}")
        print(f"Density: {density_metrics['density_percentage']:.2f}%")
        print(f"Density class: {density_metrics['density_class']}")
    
    # Visualize results
    if not args.json_output:
        print("Visualizing results...")
    result_image = visualize_results(original_image, detections, density_metrics)
    
    # Save results
    output_filename = Path(args.image).stem + "_result.jpg"
    output_path = save_results(result_image, args.output, output_filename)
    if not args.json_output:
        print(f"Results saved to: {output_path}")
    
    # Output JSON if requested
    if args.json_output:
        # Ensure output_path is absolute for the GUI
        absolute_output_path = os.path.abspath(output_path)

        # Convert potential NumPy types to standard Python types for JSON serialization
        vehicle_count = density_metrics.get('vehicle_count')
        density_percentage = density_metrics.get('density_percentage')
        density_class = density_metrics.get('density_class')

        final_metrics = {
            'vehicle_count': int(vehicle_count) if vehicle_count is not None else None,
            'density_percentage': float(density_percentage) if density_percentage is not None else None,
            'density_class': str(density_class) if density_class is not None else None, # Ensure class is string
            'output_image_path': absolute_output_path,
            'inference_time_sec': float(inference_time) # Convert inference time to float
        }
        print(json.dumps(final_metrics))

if __name__ == "__main__":
    main()