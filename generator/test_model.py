#!/usr/bin/env python3
"""
Model Testing and Inference Script for Industrial Fault Detection

This script runs inference on test images using trained YOLOv11 models
and displays/saves the results with detected industrial faults.

Usage:
    python test_model.py --model path/to/best.pt --source path/to/images
    python test_model.py --model best.pt --source test_images/ --display 10

Author: Sayan C
Project: Industrial Fault Detection using YOLOv11
"""

import os
import sys
import argparse
import glob
from pathlib import Path
import time

try:
    from ultralytics import YOLO
except ImportError:
    print("✗ Error: ultralytics package not found!")
    print("Please install it using: pip install ultralytics")
    sys.exit(1)

# Check if we're in a Jupyter/Colab environment
try:
    from IPython.display import Image, display
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    JUPYTER_ENV = True
except ImportError:
    JUPYTER_ENV = False
    try:
        import cv2
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        CV2_AVAILABLE = True
    except ImportError:
        CV2_AVAILABLE = False


def validate_model_path(model_path):
    """
    Validate that the model file exists and is a valid YOLO model.
    
    Args:
        model_path (str): Path to the model file
        
    Returns:
        bool: True if model is valid, False otherwise
    """
    if not os.path.exists(model_path):
        print(f"✗ Error: Model file not found at '{model_path}'!")
        return False
    
    if not model_path.endswith('.pt'):
        print(f"✗ Error: '{model_path}' is not a PyTorch model file (.pt)!")
        return False
    
    return True


def validate_source_path(source_path):
    """
    Validate that the source path exists and contains images.
    
    Args:
        source_path (str): Path to images directory or single image
        
    Returns:
        tuple: (bool, list) - (is_valid, list_of_images)
    """
    if not os.path.exists(source_path):
        print(f"✗ Error: Source path not found at '{source_path}'!")
        return False, []
    
    # If it's a single file
    if os.path.isfile(source_path):
        if source_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')):
            return True, [source_path]
        else:
            print(f"✗ Error: '{source_path}' is not a supported image format!")
            return False, []
    
    # If it's a directory, find all images
    if os.path.isdir(source_path):
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
        images = []
        
        for ext in image_extensions:
            images.extend(glob.glob(os.path.join(source_path, ext)))
            images.extend(glob.glob(os.path.join(source_path, ext.upper())))
        
        if not images:
            print(f"✗ Error: No images found in '{source_path}'!")
            print("Supported formats: .jpg, .jpeg, .png, .bmp, .tiff, .tif")
            return False, []
        
        print(f"✓ Found {len(images)} images in '{source_path}'")
        return True, images
    
    return False, []


def run_inference(model_path, source_path, conf_threshold=0.25, save_results=True):
    """
    Run YOLOv11 inference on the specified source.
    
    Args:
        model_path (str): Path to the trained model
        source_path (str): Path to source images
        conf_threshold (float): Confidence threshold for detections
        save_results (bool): Whether to save results
        
    Returns:
        tuple: (bool, str) - (success, results_path)
    """
    try:
        print("Loading model...")
        model = YOLO(model_path)
        
        print("Running inference...")
        print(f"Source: {source_path}")
        print(f"Confidence threshold: {conf_threshold}")
        print("-" * 50)
        
        # Run inference
        results = model.predict(
            source=source_path,
            conf=conf_threshold,
            save=save_results,
            save_txt=True,  # Save labels
            save_conf=True,  # Save confidence scores
            show_labels=True,
            show_conf=True
        )
        
        # Get results directory
        results_dir = None
        if save_results and results:
            # YOLOv11 saves results in runs/detect/predict* directories
            predict_dirs = glob.glob("runs/detect/predict*")
            if predict_dirs:
                results_dir = max(predict_dirs, key=os.path.getmtime)  # Get latest
        
        print("✓ Inference completed successfully!")
        if results_dir:
            print(f"Results saved to: {results_dir}")
        
        return True, results_dir
        
    except Exception as e:
        print(f"✗ Error during inference: {e}")
        return False, None


def display_results_jupyter(results_dir, max_images=10, image_height=400):
    """
    Display results in Jupyter/Colab environment.
    
    Args:
        results_dir (str): Path to results directory
        max_images (int): Maximum number of images to display
        image_height (int): Height of displayed images
    """
    if not JUPYTER_ENV:
        print("✗ Jupyter/Colab environment not detected!")
        return
    
    if not results_dir or not os.path.exists(results_dir):
        print("✗ Results directory not found!")
        return
    
    # Find result images
    image_patterns = ['*.jpg', '*.jpeg', '*.png']
    result_images = []
    
    for pattern in image_patterns:
        result_images.extend(glob.glob(os.path.join(results_dir, pattern)))
    
    if not result_images:
        print("✗ No result images found!")
        return
    
    print(f"Displaying {min(len(result_images), max_images)} results:")
    print("=" * 50)
    
    for i, image_path in enumerate(result_images[:max_images]):
        print(f"\nResult {i+1}: {os.path.basename(image_path)}")
        display(Image(filename=image_path, height=image_height))
        print()


def display_results_matplotlib(results_dir, max_images=10, grid_cols=2):
    """
    Display results using matplotlib for non-Jupyter environments.
    
    Args:
        results_dir (str): Path to results directory
        max_images (int): Maximum number of images to display
        grid_cols (int): Number of columns in the grid
    """
    if not results_dir or not os.path.exists(results_dir):
        print("✗ Results directory not found!")
        return
    
    # Find result images
    image_patterns = ['*.jpg', '*.jpeg', '*.png']
    result_images = []
    
    for pattern in image_patterns:
        result_images.extend(glob.glob(os.path.join(results_dir, pattern)))
    
    if not result_images:
        print("✗ No result images found!")
        return
    
    images_to_show = result_images[:max_images]
    num_images = len(images_to_show)
    
    if num_images == 0:
        return
    
    # Calculate grid dimensions
    grid_rows = (num_images + grid_cols - 1) // grid_cols
    
    # Create figure
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(15, 5*grid_rows))
    
    if grid_rows == 1:
        axes = [axes] if grid_cols == 1 else axes
    else:
        axes = axes.flatten() if num_images > 1 else [axes]
    
    for i, image_path in enumerate(images_to_show):
        try:
            img = mpimg.imread(image_path)
            axes[i].imshow(img)
            axes[i].set_title(os.path.basename(image_path))
            axes[i].axis('off')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
    
    # Hide empty subplots
    for i in range(num_images, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def print_detection_summary(results_dir):
    """
    Print summary of detections from saved label files.
    
    Args:
        results_dir (str): Path to results directory
    """
    if not results_dir or not os.path.exists(results_dir):
        return
    
    # Look for label files
    labels_dir = os.path.join(results_dir, "labels")
    if not os.path.exists(labels_dir):
        return
    
    label_files = glob.glob(os.path.join(labels_dir, "*.txt"))
    
    if not label_files:
        print("No detection labels found.")
        return
    
    total_detections = 0
    detection_counts = {}
    
    for label_file in label_files:
        with open(label_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.strip():
                    class_id = int(line.split()[0])
                    detection_counts[class_id] = detection_counts.get(class_id, 0) + 1
                    total_detections += 1
    
    print(f"\nDetection Summary:")
    print("-" * 30)
    print(f"Total detections: {total_detections}")
    print(f"Images processed: {len(label_files)}")
    
    if detection_counts:
        print("\nDetections per class:")
        for class_id, count in sorted(detection_counts.items()):
            print(f"  Class {class_id}: {count} detections")


def find_model_file():
    """
    Try to find model file in common locations.
    
    Returns:
        str: Path to model file if found, None otherwise
    """
    possible_paths = [
        'best.pt',
        'models/best.pt',
        'runs/detect/train/weights/best.pt',
        'runs/detect/train*/weights/best.pt',
        '/content/best.pt',
        'weights/best.pt'
    ]
    
    for path_pattern in possible_paths:
        if '*' in path_pattern:
            matches = glob.glob(path_pattern)
            if matches:
                return max(matches, key=os.path.getmtime)  # Return most recent
        elif os.path.exists(path_pattern):
            return path_pattern
    
    return None


def main():
    """Main function to handle command line arguments and run inference."""
    parser = argparse.ArgumentParser(
        description="Test trained YOLOv11 model on images for industrial fault detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python test_model.py --model best.pt --source test_images/
    python test_model.py --model models/best.pt --source single_image.jpg
    python test_model.py --source test_data/ --display 20 --conf 0.3
    python test_model.py --model best.pt --source webcam --no-save
        """
    )
    
    parser.add_argument(
        "--model",
        type=str,
        help="Path to trained model file (.pt) - auto-detected if not provided"
    )
    
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to source images (directory, single image, or 'webcam')"
    )
    
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for detections (default: 0.25)"
    )
    
    parser.add_argument(
        "--display",
        type=int,
        default=10,
        help="Number of result images to display (default: 10)"
    )
    
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save inference results to disk"
    )
    
    parser.add_argument(
        "--height",
        type=int,
        default=400,
        help="Height of displayed images in pixels (Jupyter only, default: 400)"
    )
    
    args = parser.parse_args()
    
    print("Industrial Fault Detection - Model Testing")
    print("=" * 50)
    
    # Find model file if not specified
    model_path = args.model
    if not model_path:
        print("Searching for trained model...")
        model_path = find_model_file()
        if not model_path:
            print("✗ Could not find trained model!")
            print("Please specify model path using --model or ensure best.pt exists in:")
            print("  - Current directory")
            print("  - models/ subdirectory")
            print("  - Training output directory")
            sys.exit(1)
        print(f"✓ Found model: {model_path}")
    
    # Validate inputs
    if not validate_model_path(model_path):
        sys.exit(1)
    
    if args.source.lower() != 'webcam':
        is_valid, image_list = validate_source_path(args.source)
        if not is_valid:
            sys.exit(1)
    
    # Run inference
    print(f"\nStarting inference...")
    success, results_dir = run_inference(
        model_path, 
        args.source, 
        args.conf, 
        not args.no_save
    )
    
    if not success:
        sys.exit(1)
    
    # Display results
    if results_dir and args.display > 0:
        print(f"\nDisplaying results...")
        
        if JUPYTER_ENV:
            display_results_jupyter(results_dir, args.display, args.height)
        else:
            display_results_matplotlib(results_dir, args.display)
    
    # Print detection summary
    if results_dir:
        print_detection_summary(results_dir)
    
    print("\n" + "=" * 50)
    print("Testing completed!")
    if results_dir:
        print(f"Results available in: {results_dir}")


if __name__ == "__main__":
    main()
