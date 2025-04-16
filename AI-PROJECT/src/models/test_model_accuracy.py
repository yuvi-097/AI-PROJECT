import os
import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.metrics import precision_score, recall_score, f1_score

def evaluate_images(test_dir, output_dir='evaluation_results'):
    """Evaluate test images and calculate metrics"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get test images
    test_images = []
    images_dir = os.path.join(test_dir, 'images')
    for img_name in os.listdir(images_dir):
        if img_name.endswith(('.jpg', '.jpeg', '.png')):
            test_images.append(os.path.join(images_dir, img_name))
    
    print(f"Found {len(test_images)} test images")
    
    # Process each image
    results = []
    for img_path in test_images:
        print(f"\nProcessing {os.path.basename(img_path)}...")
        
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            print(f"Error: Could not load image {img_path}")
            continue
            
        # Get ground truth labels if available
        label_path = os.path.join(test_dir, 'labels', 
                                os.path.splitext(os.path.basename(img_path))[0] + '.txt')
        
        gt_boxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    class_id, x_center, y_center, width, height = map(float, line.strip().split())
                    # Convert normalized coordinates to pixel coordinates
                    img_height, img_width = image.shape[:2]
                    x = int((x_center - width/2) * img_width)
                    y = int((y_center - height/2) * img_height)
                    w = int(width * img_width)
                    h = int(height * img_height)
                    gt_boxes.append([x, y, w, h])
        
        # Save visualization with ground truth boxes
        output_img = image.copy()
        for box in gt_boxes:
            x, y, w, h = box
            cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        output_path = os.path.join(output_dir,
                                f"{os.path.splitext(os.path.basename(img_path))[0]}_gt.jpg")
        cv2.imwrite(output_path, output_img)
        
        results.append({
            'image': os.path.basename(img_path),
            'gt_boxes': len(gt_boxes)
        })
    
    return results

def save_results(results, output_file):
    """Save evaluation results to file"""
    with open(output_file, 'w') as f:
        f.write("Image,Ground Truth Boxes\n")
        for r in results:
            f.write(f"{r['image']},{r['gt_boxes']}\n")

def evaluate_images_with_predictions(model_path, test_dir, output_dir='evaluation_results'):
    """Evaluate test images, calculate metrics, and compare predictions with ground truth"""
    # Load the trained YOLO model
    model = YOLO(model_path)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get test images
    test_images = []
    images_dir = os.path.join(test_dir, 'images')
    for img_name in os.listdir(images_dir):
        if img_name.endswith(('.jpg', '.jpeg', '.png')):
            test_images.append(os.path.join(images_dir, img_name))

    print(f"Found {len(test_images)} test images")

    # Initialize metrics
    all_gt_boxes = []
    all_pred_boxes = []

    # Process each image
    for img_path in test_images:
        print(f"\nProcessing {os.path.basename(img_path)}...")

        # Load image
        image = cv2.imread(img_path)
        if image is None:
            print(f"Error: Could not load image {img_path}")
            continue

        # Get ground truth labels
        label_path = os.path.join(test_dir, 'labels', \
                                os.path.splitext(os.path.basename(img_path))[0] + '.txt')

        gt_boxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    class_id, x_center, y_center, width, height = map(float, line.strip().split())
                    img_height, img_width = image.shape[:2]
                    x = int((x_center - width/2) * img_width)
                    y = int((y_center - height/2) * img_height)
                    w = int(width * img_width)
                    h = int(height * img_height)
                    gt_boxes.append([x, y, w, h])

        # Get predictions from the model
        results = model(img_path)

        # Adjusted to handle the list structure of results
        pred_boxes = []
        for result in results:  # Iterate over the list of results
            for box in result.boxes.xyxy.tolist():  # Access the bounding boxes
                x1, y1, x2, y2 = box[:4]  # Extract coordinates
                pred_boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])

        # Append to metrics lists
        all_gt_boxes.append(len(gt_boxes))
        all_pred_boxes.append(len(pred_boxes))

        # Save visualization with predictions and ground truth
        output_img = image.copy()
        for box in gt_boxes:
            x, y, w, h = box
            cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green for ground truth
        for box in pred_boxes:
            x, y, w, h = box
            cv2.rectangle(output_img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue for predictions

        output_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(img_path))[0]}_eval.jpg")
        cv2.imwrite(output_path, output_img)

    # Calculate metrics
    precision = precision_score(all_gt_boxes, all_pred_boxes, average='macro')
    recall = recall_score(all_gt_boxes, all_pred_boxes, average='macro')
    f1 = f1_score(all_gt_boxes, all_pred_boxes, average='macro')

    print(f"\nPrecision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

    return precision, recall, f1

def main():
    # Configuration
    model_path = "yolov8n.pt"  # Path to the trained YOLO model
    test_dir = "data/augmented_dataset/test"

    # Run evaluation
    print("Starting evaluation with predictions...")
    precision, recall, f1 = evaluate_images_with_predictions(model_path, test_dir)

    print("\nEvaluation complete!")

if __name__ == "__main__":
    main()