# Traffic Vehicle Density Detection

This project implements a vehicle density detection system using OpenCV and YOLO (You Only Look Once). It analyzes images of traffic scenes, detects vehicles, calculates density metrics, and visualizes the results. The system is also capable of detecting vehicles in blurry and noisy images.

**NEW**: Includes a graphical user interface (GUI) for easier interaction.

## Features

- Vehicle detection using YOLOv8
- Traffic density calculation
- Density classification (Low, Medium, High)
- Visualization with bounding boxes and density information
- Command-line interface for scripting
- **Graphical User Interface (GUI)** for interactive use
- Blurry and noisy image detection (via command-line script)
- Data preprocessing and augmentation (via command-line script)
- Model evaluation on different image conditions (via command-line script)

## Requirements

- Python 3.7+ (CustomTkinter requires >= 3.7)
- OpenCV
- PyTorch
- Ultralytics YOLOv8
- NumPy
- Matplotlib
- Pandas
- tqdm
- **CustomTkinter** (for GUI)
- **Pillow** (for GUI Image Handling)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yuvi-097/AI-PROJECT.git
   cd AI-PROJECT
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

There are two ways to use the detection feature:

### 1. Graphical User Interface (GUI)

For interactive use, run the GUI application:

```bash
python3 gui_app.py
```

This interface allows you to:
- Select the input image.
- View the input and output (annotated) images.
- See the calculated density metrics (Count, Percentage, Class).
- Configure options like the output directory, YOLO model, confidence threshold, and processing device.

### 2. Command Line Interface (CLI)

For scripting or running detection without the GUI:

```bash
python3 AI-PROJECT/traffic_density.py --image path/to/traffic_image.jpg --output results/
```

**Note:** The command-line script `AI-PROJECT/traffic_density.py` now supports a `--json_output` flag. When used, it suppresses normal console output and prints a JSON object containing the metrics and output image path to standard output. This is used internally by the GUI.

### Blurry Detection Testing

Test the model's performance on blurry and noisy images:

```
python3 test_blurry_detection.py --model runs/train/vehicle_blur_detection3/weights/best.pt --image path/to/test_image.jpg --output blurry_test_results
```

### Data Preprocessing

Preprocess and augment your dataset:

```
python3 data_preprocessing.py --input data/raw_images --output data/processed_images --annotations data/annotations.csv
```

### Command-line Arguments

#### Traffic Density Detection (`AI-PROJECT/traffic_density.py`)
- `--image`: Path to the traffic image (required)
- `--output`: Directory to save results (default: "results")
- `--model`: Path to YOLO model (default: "yolov8n.pt")
- `--conf`: Confidence threshold for detections (default: 0.25)
- `--device`: Device to run inference on ("cpu" or CUDA device number, e.g., "0")
- `--json_output`: (Flag) If present, suppresses normal output and prints results as JSON to stdout.

#### Blurry Detection Testing
- `--model`: Path to trained model weights (required)
- `--image`: Path to test image (required)
- `--conf`: Confidence threshold (default: 0.25)
- `--output`: Output directory (default: 'results')
- `--blur-level`: Blur kernel size (default: 9)
- `--salt-prob`: Salt noise probability (default: 0.02)
- `--pepper-prob`: Pepper noise probability (default: 0.02)

#### Data Preprocessing
- `--input`: Input directory with raw images
- `--output`: Output directory for processed images
- `--annotations`: Path to CSV annotations file
- `--split`: Train/val/test split ratios (default: 0.7/0.15/0.15)

## How It Works

1. **Image Loading and Preprocessing**: The input image is loaded and preprocessed for YOLO inference.
2. **Vehicle Detection**: YOLOv8 detects vehicles in the image (cars, trucks, buses, motorcycles).
3. **Density Calculation**: The system calculates density metrics based on the number and size of detected vehicles.
4. **Visualization**: Results are visualized with bounding boxes and density information.
5. **Output**: The annotated image is saved to the specified output directory.

### Blurry and Noisy Image Detection

The system can also:
1. Create variations of test images with blur and noise
2. Run detection on each variation
3. Compare detection results across different image conditions
4. Generate a summary of detection performance

### Data Preprocessing

The preprocessing pipeline includes:
1. Converting CSV annotations to YOLO format
2. Adding noise and blur to images for augmentation
3. Splitting the dataset into training, validation, and test sets
4. Organizing the processed data for training

## Density Metrics

- **Vehicle Count**: Number of vehicles detected in the image
- **Density Percentage**: Percentage of the image area occupied by vehicles
- **Density Class**: Classification of density (Low, Medium, High)

## Model Training

Train a custom YOLO model on your dataset:

```
python3 train_model.py --data vehicle_detection.yaml --epochs 100 --batch-size 16
```

## Codebase Structure

```
AI-PROJECT/
    ├── requirements.txt
    ├── traffic_density.py
    ├── assets/
    │   ├── images.jpg
    │   ├── imrs.webp
    │   └── truck.jpg
    ├── configs/
    │   └── vehicle_detection.yaml
    ├── data/
    │   ├── processed/
    │   │   ├── augmented_dataset/
    │   │   └── yolo_labels/
    │   └── raw/
    │       └── ...
    ├── results/
    │   ├── images_result.jpg
    │   ├── imrs_result.jpg
    │   ├── traffic_example_result.jpg
    │   ├── truck_result.jpg
    │   ├── blurry_test_results/
    │   │   ├── blurry_noisy.jpg
    │   │   └── ...
    │   ├── comparison/
    │   ├── detection_results/
    │   └── evaluation_results/
    ├── sample/
    │   ├── example.py
    │   └── temp
    ├── src/
    │   ├── analysis/
    │   ├── models/
    │   └── utils/
    └── tests/
        ├── test_example.py
        └── samples/

blurry_test_results2/
    ├── blurry_noisy.jpg
    ├── blurry.jpg
    ├── comparison.png
    ├── detection_summary.txt
    ├── noisy.jpg
    ├── original.jpg
    ├── blurry/
    │   └── blurry.jpg
    ├── blurry_noisy/
    │   └── blurry_noisy.jpg
    ├── noisy/
    │   └── noisy.jpg
    └── original/
        └── original.jpg

data/
    └── sample_submission.csv

detection_results2/
    ├── blurry_noisy.jpg
    ├── blurry.jpg
    ├── comparison.png
    ├── detection_summary.txt
    ├── noisy.jpg
    ├── original.jpg
    ├── blurry/
    │   └── blurry.jpg
    ├── blurry_noisy/
    │   └── blurry_noisy.jpg
    ├── noisy/
    │   └── noisy.jpg
    └── original/
        └── original.jpg

runs/
    ├── detect/
    │   └── val/
    └── train/
        ├── vehicle_blur_detection2/
        ├── vehicle_blur_detection3/
        └── vehicle_blur_detection32/
```

## Extending the Project

- Add video processing capability
- Implement a real-time monitoring system
- Add more sophisticated density algorithms
- Train custom YOLO models for specific vehicle types
- Improve robustness to different weather conditions
- Add support for more vehicle classes

## License

This project is licensed under the MIT License - see the LICENSE file for details.