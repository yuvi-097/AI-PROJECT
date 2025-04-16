import os
import argparse
from ultralytics import YOLO
import matplotlib.pyplot as plt
import torch

def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 model for vehicle detection with salt and pepper noise augmentation")
    parser.add_argument('--data', type=str, default='vehicle_detection.yaml', help='Path to data configuration file')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--img-size', type=int, default=640, help='Image size')
    parser.add_argument('--weights', type=str, default='yolov8n.pt', help='Initial weights path')
    parser.add_argument('--device', type=str, default='', help='Device to run on (cpu or cuda:0)')
    parser.add_argument('--project', type=str, default='runs/train', help='Project folder')
    parser.add_argument('--name', type=str, default='exp', help='Experiment name')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    
    return parser.parse_args()

def train_model(args):
    """Train YOLOv8 model with the provided arguments"""
    # Create output directory
    os.makedirs(os.path.join(args.project, args.name), exist_ok=True)
    
    # Set device
    device = args.device if args.device else ('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = YOLO(args.weights)
    
    # Train the model
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.img_size,
        batch=args.batch_size,
        patience=args.patience,
        device=device,
        project=args.project,
        name=args.name,
        lr0=0.01,
        lrf=0.001,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        save=True,
        save_period=10
    )
    
    # Validate the model on the test set
    print("\nEvaluating model on test set...")
    model.val()
    
    return model, results

def plot_results(results, output_dir):
    """Plot and save training results"""
    # Make sure the results have metrics
    if not hasattr(results, 'keys'):
        print("No metrics available for plotting")
        return
    
    # Create plots directory
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot metrics
    metrics = ['box_loss', 'cls_loss', 'dfl_loss', 'precision', 'recall', 'mAP50', 'mAP50-95']
    
    for metric in metrics:
        if metric in results.keys():
            plt.figure(figsize=(10, 6))
            plt.plot(results[metric], label=metric)
            plt.title(f'Training {metric}')
            plt.xlabel('Epoch')
            plt.ylabel(metric)
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(plots_dir, f'{metric}.png'))
            plt.close()
    
    print(f"Plots saved to {plots_dir}")

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Train the model
    print(f"Training YOLOv8 model with the following settings:")
    print(f"- Data config: {args.data}")
    print(f"- Initial weights: {args.weights}")
    print(f"- Epochs: {args.epochs}")
    print(f"- Batch size: {args.batch_size}")
    print(f"- Image size: {args.img_size}")
    print(f"- Output: {os.path.join(args.project, args.name)}")
    
    model, results = train_model(args)
    
    # Plot and save results
    output_dir = os.path.join(args.project, args.name)
    plot_results(results, output_dir)
    
    print(f"\nTraining complete. Model saved to {output_dir}")
    print("To use the trained model for inference, run:")
    print(f"from ultralytics import YOLO; model = YOLO('{os.path.join(output_dir, 'weights/best.pt')}')")

if __name__ == "__main__":
    main() 