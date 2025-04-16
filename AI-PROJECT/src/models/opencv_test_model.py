import os
import cv2
import numpy as np

def load_model(weights_path, cfg_path):
    """Load YOLO model using OpenCV's DNN module"""
    net = cv2.dnn.readNet(weights_path, cfg_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net

def detect_objects(net, image, conf_threshold=0.25, nms_threshold=0.45):
    """Detect objects in an image using OpenCV DNN"""
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Create blob from image
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (640, 640), swapRB=True, crop=False)
    net.setInput(blob)
    
    # Get output layer names
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    
    # Forward pass
    outputs = net.forward(output_layers)
    
    # Process detections
    boxes = []
    confidences = []
    class_ids = []
    
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > conf_threshold:
                # Scale bounding box coordinates back relative to image size
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                # Rectangle coordinates
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Apply non-maximum suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    
    detections = []
    if len(indices) > 0:
        indices = indices.flatten()
        for i in indices:
            detections.append({
                'box': boxes[i],
                'confidence': confidences[i],
                'class_id': class_ids[i]
            })
    
    return detections

def evaluate_model(model_path, cfg_path, test_dir, conf_threshold=0.25):
    """Evaluate model on test images"""
    # Load model
    print(f"Loading model from {model_path}...")
    net = load_model(model_path, cfg_path)
    
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
        
        # Load and process image
        image = cv2.imread(img_path)
        if image is None:
            print(f"Error: Could not load image {img_path}")
            continue
        
        # Run detection
        detections = detect_objects(net, image, conf_threshold)
        
        # Calculate metrics
        confidences = [d['confidence'] for d in detections]
        avg_conf = np.mean(confidences) if confidences else 0
        max_conf = np.max(confidences) if confidences else 0
        
        print(f"- Detections: {len(detections)}")
        print(f"- Average confidence: {avg_conf:.3f}")
        print(f"- Maximum confidence: {max_conf:.3f}")
        
        # Save visualization
        output_img = image.copy()
        for det in detections:
            x, y, w, h = det['box']
            cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(output_img, f"{det['confidence']:.2f}", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        output_path = os.path.join('evaluation_results',
                                 f"{os.path.splitext(os.path.basename(img_path))[0]}_result.jpg")
        cv2.imwrite(output_path, output_img)
        
        results.append({
            'image': os.path.basename(img_path),
            'detections': len(detections),
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
    weights_path = "runs/train/vehicle_blur_detection/weights/best.pt"
    cfg_path = "models/yolov8n.yaml"  # You'll need to provide the correct path to the model config
    test_dir = "data/augmented_dataset/test"
    output_file = "model_evaluation_results.csv"
    conf_threshold = 0.25
    
    # Create output directory
    os.makedirs('evaluation_results', exist_ok=True)
    
    # Run evaluation
    print("Starting model evaluation...")
    results = evaluate_model(weights_path, cfg_path, test_dir, conf_threshold)
    
    # Save results
    save_results(results, output_file)
    print(f"\nEvaluation complete! Results saved to {output_file}")

if __name__ == "__main__":
    main() 