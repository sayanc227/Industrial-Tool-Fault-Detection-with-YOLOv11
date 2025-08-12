#!/usr/bin/env python3
"""
YOLOv11 Training Script for Industrial Fault Detection

This script handles the training of YOLOv11 models for industrial fault detection.
It provides configurable parameters for training epochs, image size, model variant,
and other hyperparameters.

Usage:
    python run_inference.py [--data path/to/data.yaml] [--model yolo11s.pt] [--epochs 100]

Author: Sayan Chatterjee
Project: Industrial Fault Detection using YOLOv11
"""

import os
import sys
import argparse
import time
from datetime import datetime
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("✗ Error: ultralytics package not found!")
    print("Please install it using: pip install ultralytics")
    sys.exit(1)


def validate_data_yaml(data_path):
    """
    Validate that the data.yaml file exists and is readable.
    
    Args:
        data_path (str): Path to the data.yaml file
        
    Returns:
        bool: True if file is valid, False otherwise
    """
    if not os.path.exists(data_path):
        print(f"✗ Error: Data configuration file not found at '{data_path}'!")
        print("Please run create_yaml.py first to generate the configuration file.")
        return False
    
    if not data_path.endswith(('.yaml', '.yml')):
        print(f"✗ Error: '{data_path}' is not a YAML file!")
        return False
    
    return True


def setup_training_directory():
    """
    Create directories for training outputs.
    
    Returns:
        str: Path to the runs directory
    """
    runs_dir = "runs/detect"
    os.makedirs(runs_dir, exist_ok=True)
    
    # Create timestamped subdirectory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_dir = os.path.join(runs_dir, f"train_{timestamp}")
    os.makedirs(train_dir, exist_ok=True)
    
    return train_dir


def get_available_models():
    """
    Get list of available YOLOv11 model variants.
    
    Returns:
        list: Available model variants
    """
    return ['yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt', 'yolo11l.pt', 'yolo11x.pt']


def print_training_info(args, train_dir):
    """
    Print training configuration information.
    
    Args:
        args: Command line arguments
        train_dir (str): Training output directory
    """
    print("Training Configuration:")
    print("-" * 40)
    print(f"Data config: {args.data}")
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Image size: {args.imgsz}")
    print(f"Batch size: {args.batch}")
    print(f"Workers: {args.workers}")
    print(f"Device: {args.device}")
    print(f"Output directory: {train_dir}")
    print(f"Save period: Every {args.save_period} epochs")
    print("-" * 40)


def train_model(args, train_dir):
    """
    Execute YOLOv11 model training.
    
    Args:
        args: Command line arguments containing training parameters
        train_dir (str): Directory for training outputs
        
    Returns:
        bool: True if training completed successfully, False otherwise
    """
    try:
        print("Starting YOLOv11 training...")
        print("This may take a while depending on your dataset size and hardware.")
        print("-" * 60)
        
        # Initialize YOLO model
        model = YOLO(args.model)
        
        # Start training
        start_time = time.time()
        
        results = model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            device=args.device,
            project=os.path.dirname(train_dir),
            name=os.path.basename(train_dir),
            save_period=args.save_period,
            patience=args.patience,
            save=True,
            plots=True,
            val=True
        )
        
        end_time = time.time()
        training_time = end_time - start_time
        
        print("\n" + "=" * 60)
        print("✓ Training completed successfully!")
        print(f"Training time: {training_time/3600:.2f} hours")
        print(f"Results saved to: {train_dir}")
        
        # Print training summary
        if results:
            print("\nTraining Summary:")
            print("-" * 30)
            print(f"Best mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
            print(f"Best mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error during training: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check if your data.yaml file is correct")
        print("2. Ensure your dataset is properly formatted")
        print("3. Verify you have enough disk space")
        print("4. Check GPU memory if using CUDA")
        return False


def find_data_yaml():
    """
    Try to find data.yaml file in common locations.
    
    Returns:
        str: Path to data.yaml file if found, None otherwise
    """
    possible_paths = [
        'data.yaml',
        'config/data.yaml',
        '../data.yaml',
        '/content/data.yaml',
        './Industrial_Faults_Detection_Dataset/data.yaml'
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None


def main():
    """Main function to handle command line arguments and execute training."""
    parser = argparse.ArgumentParser(
        description="Train YOLOv11 model for industrial fault detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_inference.py
    python run_inference.py --epochs 200 --imgsz 832
    python run_inference.py --model yolo11m.pt --batch 16
    python run_inference.py --data custom_data.yaml --device cpu
        """
    )
    
    parser.add_argument(
        "--data",
        type=str,
        help="Path to data.yaml file (auto-detected if not provided)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11s.pt",
        choices=get_available_models(),
        help="YOLOv11 model variant (default: yolo11s.pt)"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)"
    )
    
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size for training (default: 640)"
    )
    
    parser.add_argument(
        "--batch",
        type=int,
        default=-1,
        help="Batch size, -1 for auto-batch (default: -1)"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of worker threads (default: 8)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Training device: '', 'cpu', '0', '0,1', etc. (default: auto-detect)"
    )
    
    parser.add_argument(
        "--save_period",
        type=int,
        default=10,
        help="Save model every N epochs (default: 10)"
    )
    
    parser.add_argument(
        "--patience",
        type=int,
        default=50,
        help="Early stopping patience in epochs (default: 50)"
    )
    
    args = parser.parse_args()
    
    print("Industrial Fault Detection - YOLOv11 Training")
    print("=" * 50)
    
    # Determine data.yaml path
    data_path = args.data
    if not data_path:
        print("Searching for data.yaml file...")
        data_path = find_data_yaml()
        if not data_path:
            print("✗ Could not find data.yaml file!")
            print("Please specify the path using --data or ensure data.yaml exists in:")
            print("  - Current directory")
            print("  - config/ subdirectory")
            print("  - Dataset directory")
            sys.exit(1)
        print(f"✓ Found data config: {data_path}")
    
    # Validate data configuration
    if not validate_data_yaml(data_path):
        sys.exit(1)
    
    args.data = data_path
    
    # Setup training directory
    train_dir = setup_training_directory()
    
    # Print training information
    print_training_info(args, train_dir)
    
    # Confirm before starting training
    if args.epochs > 50:
        response = input(f"\nThis will train for {args.epochs} epochs. Continue? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("Training cancelled.")
            sys.exit(0)
    
    # Execute training
    success = train_model(args, train_dir)
    
    if success:
        print("\n" + "=" * 50)
        print("Training completed successfully!")
        print("Check the results directory for:")
        print("  - Trained model weights (best.pt, last.pt)")
        print("  - Training metrics and plots")
        print("  - Validation results")
    else:
        print("\n" + "=" * 50)
        print("Training failed. Please check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
