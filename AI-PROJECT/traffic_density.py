
import argparse
import os
import time
import cv2
import numpy as np
from pathlib import Path
import torch
from ultralytics import YOLO

from utils import (
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
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading YOLO model: {args.model}")
    model = load_model(args.model, device)
    
    # Load and preprocess image
    print(f"Processing image: {args.image}")
    original_image = load_image(args.image)
    preprocessed_image, original_image, image_shape = preprocess_image(original_image)
    
    # Detect vehicles
    print("Detecting vehicles...")
    start_time = time.time()
    detections = detect_vehicles(model, original_image, args.conf)
    inference_time = time.time() - start_time
    print(f"Detection completed in {inference_time:.2f} seconds")
    
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
    output_filename = Path(args.image).stem + "_result.jpg"
    output_path = save_results(result_image, args.output, output_filename)
    print(f"Results saved to: {output_path}")

if __name__ == "__main__":
    main() 