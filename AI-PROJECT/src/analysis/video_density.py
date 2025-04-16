"""
Video version of the traffic density detection system.
This script processes videos frame by frame and outputs an annotated video.
"""
import argparse
import os
import time
import cv2
import numpy as np
from pathlib import Path
import torch
from tqdm import tqdm
from ultralytics import YOLO

from utils import (
    preprocess_image,
    calculate_density,
    visualize_results
)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Traffic Vehicle Density Detection for Videos")
    parser.add_argument("--video", required=True, help="Path to the traffic video")
    parser.add_argument("--output", default="results", help="Output directory for results")
    parser.add_argument("--model", default="yolov8n.pt", help="Path to YOLO model")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--device", default="", help="Device to run inference on (cuda device, i.e. 0 or cpu)")
    parser.add_argument("--skip", type=int, default=1, help="Process every nth frame (default: 1)")
    parser.add_argument("--display", action="store_true", help="Display frames while processing")
    
    return parser.parse_args()

def load_model(model_path, device=""):
    """Load YOLO model"""
    # Download the model if it doesn't exist and it's a standard YOLO model name
    if not os.path.exists(model_path) and model_path.startswith("yolov8"):
        print(f"Downloading {model_path}...")
    
    # Load the model
    model = YOLO(model_path)
    
    # Set device if specified
    if device:
        model.to(device)
        
    return model

def detect_vehicles(model, frame, conf_threshold=0.25):
    """Detect vehicles in a video frame using YOLO"""
    # Run inference
    results = model(frame, conf=conf_threshold)[0]
    
    # Extract detections
    detections = []
    for det in results.boxes.data.cpu().numpy():
        x1, y1, x2, y2, conf, class_id = det
        detections.append([x1, y1, x2, y2, conf, class_id])
    
    return detections

def process_video(video_path, output_dir, model, conf_threshold=0.25, skip_frames=1, display=False, device="cpu"):
    """Process a video for traffic density detection"""
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create output video writer
    output_path = os.path.join(output_dir, Path(video_path).stem + "_result.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Initialize density tracking
    frame_count = 0
    processed_frames = 0
    density_history = []
    
    # Process video
    pbar = tqdm(total=total_frames)
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        pbar.update(1)
        
        # Skip frames if needed
        if frame_count % skip_frames != 0:
            continue
        
        processed_frames += 1
        
        # Preprocess frame
        _, _, image_shape = preprocess_image(frame)
        
        # Detect vehicles
        detections = detect_vehicles(model, frame, conf_threshold)
        
        # Calculate density
        density_metrics = calculate_density(detections, image_shape)
        density_history.append(density_metrics)
        
        # Visualize results
        result_frame = visualize_results(frame, detections, density_metrics)
        
        # Add frame number
        cv2.putText(result_frame, f"Frame: {frame_count}", (10, height - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Write frame to output video
        out.write(result_frame)
        
        # Display frame if requested
        if display:
            cv2.imshow("Traffic Density Detection", result_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Calculate processing statistics
    processing_time = time.time() - start_time
    fps_processing = processed_frames / processing_time if processing_time > 0 else 0
    
    # Calculate average density
    avg_density = sum(d["density_percentage"] for d in density_history) / len(density_history) if density_history else 0
    avg_vehicle_count = sum(d["vehicle_count"] for d in density_history) / len(density_history) if density_history else 0
    
    # Clean up
    cap.release()
    out.release()
    if display:
        cv2.destroyAllWindows()
    pbar.close()
    
    print(f"\nVideo processing complete:")
    print(f"- Total frames: {total_frames}")
    print(f"- Processed frames: {processed_frames}")
    print(f"- Processing time: {processing_time:.2f} seconds")
    print(f"- Processing speed: {fps_processing:.2f} fps")
    print(f"- Average vehicle count: {avg_vehicle_count:.2f}")
    print(f"- Average density: {avg_density:.2f}%")
    print(f"- Output saved to: {output_path}")
    
    return output_path, density_history

def main():
    # Parse arguments
    args = parse_arguments()
    
    # Set device
    device = args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading YOLO model: {args.model}")
    model = load_model(args.model, device)
    
    # Process video
    print(f"Processing video: {args.video}")
    output_path, density_history = process_video(
        args.video, 
        args.output, 
        model, 
        args.conf, 
        args.skip, 
        args.display, 
        device
    )

if __name__ == "__main__":
    main() 