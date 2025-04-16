import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def load_image(image_path):
    """
    Load an image from the given path
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        np.ndarray: Loaded image in BGR format
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Unable to read image at {image_path}")
    
    return img

def preprocess_image(image, target_size=(640, 640)):
    """
    Preprocess image for YOLO inference
    
    Args:
        image (np.ndarray): Input image in BGR format
        target_size (tuple): Target size for resizing (width, height)
        
    Returns:
        tuple: (preprocessed_image, original_image, (original_height, original_width))
    """
    original_image = image.copy()
    height, width = image.shape[:2]
    
    # Resize image while maintaining aspect ratio
    resized_image = cv2.resize(image, target_size)
    
    # Normalize pixel values (0-255 to 0-1)
    normalized_image = resized_image / 255.0
    
    return normalized_image, original_image, (height, width)

def calculate_density(detections, image_shape, vehicle_classes=[2, 3, 5, 7]):
    """
    Calculate vehicle density based on detections
    
    Args:
        detections (list): List of detections from YOLO
        image_shape (tuple): Original image shape (height, width)
        vehicle_classes (list): List of class IDs representing vehicles
                               Default: [2, 3, 5, 7] for 'car', 'motorcycle', 'bus', 'truck' in COCO
    
    Returns:
        dict: Dictionary containing density metrics
    """
    height, width = image_shape
    image_area = height * width
    
    # Filter vehicle detections
    vehicle_detections = [det for det in detections if int(det[5]) in vehicle_classes]
    vehicle_count = len(vehicle_detections)
    
    # Calculate total area occupied by vehicles
    vehicle_area = 0
    for det in vehicle_detections:
        x1, y1, x2, y2 = det[0], det[1], det[2], det[3]
        vehicle_area += (x2 - x1) * (y2 - y1)
    
    # Calculate density metrics
    density_percentage = (vehicle_area / image_area) * 100 if image_area > 0 else 0
    vehicles_per_area = vehicle_count / (image_area / 10000)  # vehicles per 10000 square pixels
    
    # Classify density
    if density_percentage < 5:
        density_class = "Low"
    elif density_percentage < 15:
        density_class = "Medium"
    else:
        density_class = "High"
    
    return {
        "vehicle_count": vehicle_count,
        "density_percentage": density_percentage,
        "vehicles_per_area": vehicles_per_area,
        "density_class": density_class
    }

def visualize_results(image, detections, density_metrics, vehicle_classes=[2, 3, 5, 7], class_names=None):
    """
    Visualize detection results and density metrics
    
    Args:
        image (np.ndarray): Original image
        detections (list): List of detections from YOLO
        density_metrics (dict): Dictionary containing density metrics
        vehicle_classes (list): List of class IDs representing vehicles
        class_names (list): List of class names corresponding to class IDs
        
    Returns:
        np.ndarray: Visualized image
    """
    result_img = image.copy()
    
    # Default COCO class names for vehicles if not provided
    if class_names is None:
        class_names = ["", "", "car", "motorcycle", "", "bus", "", "truck"]
    
    # Colors for different vehicle types (BGR format)
    colors = {
        2: (0, 255, 0),    # Car: Green
        3: (255, 0, 0),    # Motorcycle: Blue
        5: (0, 0, 255),    # Bus: Red
        7: (255, 255, 0)   # Truck: Cyan
    }
    
    # Draw bounding boxes
    for det in detections:
        x1, y1, x2, y2 = int(det[0]), int(det[1]), int(det[2]), int(det[3])
        conf = det[4]
        class_id = int(det[5])
        
        if class_id in vehicle_classes:
            color = colors.get(class_id, (0, 255, 0))
            cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
            
            label = f"{class_names[class_id]}: {conf:.2f}"
            cv2.putText(result_img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Draw density information
    density_text = [
        f"Vehicle Count: {density_metrics['vehicle_count']}",
        f"Density: {density_metrics['density_percentage']:.2f}%",
        f"Class: {density_metrics['density_class']}"
    ]
    
    y_pos = 30
    for text in density_text:
        cv2.putText(result_img, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 5)
        cv2.putText(result_img, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_pos += 30
    
    return result_img

def save_results(image, output_dir, filename):
    """
    Save the result image to the output directory
    
    Args:
        image (np.ndarray): Result image to save
        output_dir (str): Output directory path
        filename (str): Output filename
        
    Returns:
        str: Path to the saved image
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, image)
    
    return output_path 