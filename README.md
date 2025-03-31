# Traffic Vehicle Density Detection

This project implements a vehicle density detection system using OpenCV and YOLO (You Only Look Once). It analyzes images of traffic scenes, detects vehicles, calculates density metrics, and visualizes the results.

## Features

- Vehicle detection using YOLOv8
- Traffic density calculation
- Density classification (Low, Medium, High)
- Visualization with bounding boxes and density information
- Command-line interface for easy usage

## Requirements

- Python 3.6+
- OpenCV
- PyTorch
- Ultralytics YOLOv8
- NumPy
- Matplotlib

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/your-username/traffic-density-detection.git
   cd traffic-density-detection
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the script with an input image:

```
python3 traffic_density.py --image path/to/traffic_image.jpg --output results/
```

### Command-line Arguments

- `--image`: Path to the traffic image (required)
- `--output`: Directory to save results (default: "results")
- `--model`: Path to YOLO model (default: "yolov8n.pt")
- `--conf`: Confidence threshold for detections (default: 0.25)
- `--device`: Device to run inference on ("cpu" or CUDA device number, e.g., "0")

## Example

```
python3 traffic_density.py --image samples/traffic.jpg --output results --model yolov8n.pt --conf 0.3
```

## How It Works

1. **Image Loading and Preprocessing**: The input image is loaded and preprocessed for YOLO inference.
2. **Vehicle Detection**: YOLOv8 detects vehicles in the image (cars, trucks, buses, motorcycles).
3. **Density Calculation**: The system calculates density metrics based on the number and size of detected vehicles.
4. **Visualization**: Results are visualized with bounding boxes and density information.
5. **Output**: The annotated image is saved to the specified output directory.

## Density Metrics

- **Vehicle Count**: Number of vehicles detected in the image
- **Density Percentage**: Percentage of the image area occupied by vehicles
- **Density Class**: Classification of density (Low, Medium, High)

## Extending the Project

- Add video processing capability
- Implement a real-time monitoring system
- Add more sophisticated density algorithms
- Train custom YOLO models for specific vehicle types

## License

This project is licensed under the MIT License - see the LICENSE file for details. 