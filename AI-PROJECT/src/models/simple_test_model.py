import os
from ultralytics import YOLO
import cv2
import numpy as np

def evaluate_model(model_path, test_dir, conf_threshold=0.25):
    """Evaluate model on test images"""
    # Load model
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    # Get test images
    test_images = []
    images_dir = os.path.join(test_dir, 'images')
    for img_name in os.listdir(images_dir):
        if img_name.endswith(('.jpg', '.jpeg', '.png')):
            test_images.append(os.path.join(images_dir, img_name))
    
    print(f"Found {len(test_images)} test images")
    
    # Evaluate each image
    results = []
    for img_path in test_images:
        print(f"\nProcessing {os.path.basename(img_path)}...")
        
        # Run prediction
        pred = model(img_path, conf=conf_threshold)[0]
        
        # Get detection metrics
        boxes = pred.boxes
        num_detections = len(boxes)
        confidences = boxes.conf.cpu().numpy() if num_detections > 0 else []
        
        # Calculate metrics
        avg_conf = np.mean(confidences) if len(confidences) > 0 else 0
        max_conf = np.max(confidences) if len(confidences) > 0 else 0
        
        print(f"- Detections: {num_detections}")
        print(f"- Average confidence: {avg_conf:.3f}")
        print(f"- Maximum confidence: {max_conf:.3f}")
        
        results.append({
            'image': os.path.basename(img_path),
            'detections': num_detections,
            'avg_confidence': avg_conf,
            'max_confidence': max_conf
        })
    
    return results

def save_results(results, output_file):
    """Save evaluation results to file"""
    with open(output_file, 'w') as f:
        f.write("Image,Detections,Average Confidence,Maximum Confidence\n")
        for r in results:
            f.write(f"{r['image']},{r['detections']},{r['avg_confidence']:.3f},{r['max_confidence']:.3f}\n")

def main():
    # Configuration
    model_path = "runs/train/vehicle_blur_detection/weights/best.pt"
    test_dir = "data/augmented_dataset/test"
    output_file = "model_evaluation_results.csv"
    conf_threshold = 0.25
    
    # Run evaluation
    print("Starting model evaluation...")
    results = evaluate_model(model_path, test_dir, conf_threshold)
    
    # Save results
    save_results(results, output_file)
    print(f"\nEvaluation complete! Results saved to {output_file}")

if __name__ == "__main__":
    main() 