#!/usr/bin/env python3
"""
Simple Demo Inference Script - Industrial Fault Detection

A simplified script that closely matches your original Colab workflow.
Perfect for quick testing and demonstration purposes.

Usage:
    python demo_inference.py
    python demo_inference.py --model_path /path/to/best.pt --images_path /path/to/images

Author: Sayan C 
Project: Industrial Fault Detection using YOLOv11
"""

import os
import sys
import glob
import argparse

try:
    from ultralytics import YOLO
    print("✓ YOLOv11 (ultralytics) loaded successfully")
except ImportError:
    print("✗ Error: ultralytics package not found!")
    print("Install with: pip install ultralytics")
    sys.exit(1)

# Check environment and import display libraries
try:
    # For Jupyter/Colab environments
    from IPython.display import Image, display
    JUPYTER_ENV = True
    print("✓ Jupyter/Colab environment detected")
except ImportError:
    JUPYTER_ENV = False
    print("ℹ Non-Jupyter environment detected")
    try:
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        MATPLOTLIB_AVAILABLE = True
        print("✓ Matplotlib available for display")
    except ImportError:
        MATPLOTLIB_AVAILABLE = False
        print("✗ No display libraries available")


def run_simple_inference(model_path="/content/best.pt", 
                        images_path="/content/drive/MyDrive/YOLO/Industrial Faults Detection Dataset/images"):
    """
    Run inference using YOLOv11 - matches your original Colab command.
    
    Args:
        model_path (str): Path to the trained model
        images_path (str): Path to the images directory
    """
    
    print("🚀 Starting Industrial Fault Detection Inference")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Images: {images_path}")
    print("-" * 60)
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"❌ Model not found at: {model_path}")
        print("Please check the model path or train a model first")
        return False
    
    if not os.path.exists(images_path):
        print(f"❌ Images directory not found at: {images_path}")
        print("Please check the images path")
        return False
    
    try:
        # Load model
        print("📥 Loading model...")
        model = YOLO(model_path)
        print("✅ Model loaded successfully!")
        
        # Run inference - equivalent to your YOLO command
        print("🔍 Running inference...")
        results = model.predict(
            source=images_path,
            save=True,
            conf=0.25,  # Default confidence threshold
            show_labels=True,
            show_conf=True
        )
        
        print("✅ Inference completed!")
        
        # Find the latest prediction directory
        predict_dirs = glob.glob('/content/runs/detect/predict*')
        if not predict_dirs:
            predict_dirs = glob.glob('runs/detect/predict*')
        
        if predict_dirs:
            latest_dir = max(predict_dirs, key=os.path.getmtime)
            print(f"📁 Results saved to: {latest_dir}")
            return latest_dir
        else:
            print("⚠️ Could not find results directory")
            return None
            
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        return False


def display_results(results_dir, max_images=50, image_height=400):
    """
    Display inference results - matches your original display code.
    
    Args:
        results_dir (str): Path to the results directory
        max_images (int): Maximum number of images to display
        image_height (int): Height of images in pixels (for Jupyter)
    """
    
    if not results_dir:
        print("❌ No results directory provided")
        return
    
    # Find all result images (matches your glob pattern)
    image_pattern = os.path.join(results_dir, '*.jpg')
    result_images = glob.glob(image_pattern)
    
    if not result_images:
        # Try other common formats
        for ext in ['*.png', '*.jpeg']:
            result_images.extend(glob.glob(os.path.join(results_dir, ext)))
    
    if not result_images:
        print(f"❌ No result images found in {results_dir}")
        return
    
    print(f"🖼️ Displaying {min(len(result_images), max_images)} result images:")
    print("-" * 60)
    
    # Display images (matches your original code style)
    displayed_count = 0
    for image_path in result_images[:max_images]:
        displayed_count += 1
        
        if JUPYTER_ENV:
            # Original Jupyter/Colab display method
            print(f"Result {displayed_count}: {os.path.basename(image_path)}")
            display(Image(filename=image_path, height=image_height))
            print('\n')
        else:
            # Alternative display for non-Jupyter environments
            print(f"Result {displayed_count}: {os.path.basename(image_path)}")
            if MATPLOTLIB_AVAILABLE:
                try:
                    img = mpimg.imread(image_path)
                    plt.figure(figsize=(10, 6))
                    plt.imshow(img)
                    plt.title(f"Detection Result: {os.path.basename(image_path)}")
                    plt.axis('off')
                    plt.tight_layout()
                    plt.show()
                except Exception as e:
                    print(f"Could not display image: {e}")
            else:
                print(f"  📄 Saved at: {image_path}")
    
    print(f"✅ Displayed {displayed_count} images")


def main():
    """
    Main function - recreates your original Colab workflow.
    """
    
    parser = argparse.ArgumentParser(description="Demo inference for Industrial Fault Detection")
    parser.add_argument("--model_path", type=str, default="/content/best.pt", 
                       help="Path to trained model (default: /content/best.pt)")
    parser.add_argument("--images_path", type=str, 
                       default="/content/drive/MyDrive/YOLO/Industrial Faults Detection Dataset/images",
                       help="Path to images directory")
    parser.add_argument("--max_display", type=int, default=50, 
                       help="Maximum images to display (default: 50)")
    parser.add_argument("--image_height", type=int, default=400, 
                       help="Display height in pixels (default: 400)")
    
    args = parser.parse_args()
    
    # Try to find local paths if Colab paths don't exist
    if not os.path.exists(args.model_path):
        local_model_paths = ['best.pt', 'models/best.pt', 'runs/detect/train/weights/best.pt']
        for path in local_model_paths:
            if os.path.exists(path):
                args.model_path = path
                break
    
    if not os.path.exists(args.images_path):
        local_image_paths = ['images/', 'test_images/', 'data/images/', 'validation/images/']
        for path in local_image_paths:
            if os.path.exists(path):
                args.images_path = path
                break
    
    print("🏭 Industrial Fault Detection Demo")
    print("=" * 60)
    print("This script recreates your original Colab inference workflow:")
    print("1. Load trained YOLOv11 model")
    print("2. Run inference on images")
    print("3. Display results with detections")
    print("=" * 60)
    
    # Step 1: Run inference (equivalent to your !yolo command)
    results_dir = run_simple_inference(args.model_path, args.images_path)
    
    if results_dir:
        print("\n" + "🎯 INFERENCE SUCCESSFUL!" + "\n")
        
        # Step 2: Display results (equivalent to your display loop)
        display_results(results_dir, args.max_display, args.image_height)
        
        print("\n" + "=" * 60)
        print("✅ Demo completed successfully!")
        print(f"📁 All results saved in: {results_dir}")
        
        # Additional info
        total_images = len(glob.glob(os.path.join(results_dir, '*.jpg')))
        if total_images == 0:
            total_images = len(glob.glob(os.path.join(results_dir, '*.png')))
        print(f"📊 Total images processed: {total_images}")
        
    else:
        print("\n" + "❌ DEMO FAILED!")
        print("Please check your model and image paths")


if __name__ == "__main__":
    main()


# Additional utility functions for advanced users
def batch_inference_with_stats(model_path, images_path):
    """
    Run inference and return detailed statistics.
    """
    model = YOLO(model_path)
    results = model.predict(source=images_path, save=True, verbose=True)
    
    stats = {
        'total_images': len(results),
        'total_detections': 0,
        'confidence_scores': [],
        'detection_classes': []
    }
    
    for result in results:
        if result.boxes is not None:
            stats['total_detections'] += len(result.boxes)
            if hasattr(result.boxes, 'conf'):
                stats['confidence_scores'].extend(result.boxes.conf.tolist())
            if hasattr(result.boxes, 'cls'):
                stats['detection_classes'].extend(result.boxes.cls.tolist())
    
    return stats


def create_detection_report(results_dir):
    """
    Create a simple text report of detections.
    """
    label_files = glob.glob(os.path.join(results_dir, 'labels', '*.txt'))
    
    report = []
    report.append("Industrial Fault Detection Report")
    report.append("=" * 40)
    report.append(f"Images processed: {len(label_files)}")
    
    total_detections = 0
    for label_file in label_files:
        with open(label_file, 'r') as f:
            detections = len([line for line in f if line.strip()])
            total_detections += detections
    
    report.append(f"Total detections: {total_detections}")
    report.append(f"Average detections per image: {total_detections/len(label_files) if label_files else 0:.2f}")
    
    return '\n'.join(report)
