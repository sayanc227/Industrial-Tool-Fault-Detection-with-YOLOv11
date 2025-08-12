#!/usr/bin/env python3
"""
YAML Configuration Generator for Industrial Fault Detection

This script automatically creates a data.yaml configuration file for YOLOv11 training.
It reads class names from a classes.txt file and generates the proper YAML format
with paths, number of classes, and class names.

Usage:
    python create_yaml.py [--classes_path path/to/classes.txt] [--output_path path/to/data.yaml]

Author: Sayan C 
Project: Industrial Fault Detection using YOLOv11
"""

import yaml
import os
import argparse
import sys
from pathlib import Path


def validate_classes_file(classes_path):
    """
    Validate that the classes file exists and is readable.
    
    Args:
        classes_path (str): Path to the classes.txt file
        
    Returns:
        bool: True if file is valid, False otherwise
    """
    if not os.path.exists(classes_path):
        print(f"✗ Error: classes.txt file not found at '{classes_path}'!")
        print("Please create a classes.txt file with one class name per line.")
        return False
    
    if not os.path.isfile(classes_path):
        print(f"✗ Error: '{classes_path}' is not a file!")
        return False
    
    return True


def read_classes(classes_path):
    """
    Read class names from the classes.txt file.
    
    Args:
        classes_path (str): Path to the classes.txt file
        
    Returns:
        list: List of class names
    """
    classes = []
    
    try:
        with open(classes_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f.readlines(), 1):
                line = line.strip()
                if len(line) == 0:
                    continue  # Skip empty lines
                if line.startswith('#'):
                    continue  # Skip comment lines
                classes.append(line)
                
        if not classes:
            print(f"✗ Error: No valid class names found in '{classes_path}'!")
            print("Make sure the file contains at least one class name per line.")
            return None
            
        print(f"✓ Found {len(classes)} classes: {classes}")
        return classes
        
    except Exception as e:
        print(f"✗ Error reading classes file: {e}")
        return None


def create_data_yaml(classes_path, output_path, data_root="/content/data"):
    """
    Create a YAML configuration file for YOLOv11 training.
    
    Args:
        classes_path (str): Path to the classes.txt file
        output_path (str): Path where data.yaml will be saved
        data_root (str): Root path for the dataset
    """
    print("Creating YAML configuration file...")
    print("-" * 40)
    
    # Validate and read classes
    if not validate_classes_file(classes_path):
        sys.exit(1)
    
    classes = read_classes(classes_path)
    if classes is None:
        sys.exit(1)
    
    number_of_classes = len(classes)
    
    # Create data dictionary with YOLOv11 format
    data = {
        'path': data_root,
        'train': 'train/images',
        'val': 'validation/images',
        'test': 'validation/images',  # Use validation as test if no separate test set
        'nc': number_of_classes,
        'names': classes
    }
    
    # Add metadata
    data['metadata'] = {
        'project': 'Industrial Fault Detection',
        'author': 'Sayan Chatterjee',
        'description': 'YOLOv11 configuration for industrial fault detection',
        'version': '1.0'
    }
    
    try:
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Write data to YAML file
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, sort_keys=False, default_flow_style=False, indent=2)
        
        print(f"✓ Successfully created config file at '{output_path}'")
        
        # Display file contents
        print(f"\nGenerated YAML content:")
        print("-" * 40)
        with open(output_path, 'r') as f:
            print(f.read())
            
        return True
        
    except Exception as e:
        print(f"✗ Error creating YAML file: {e}")
        return False


def find_classes_file():
    """
    Try to find classes.txt file in common locations.
    
    Returns:
        str: Path to classes.txt file if found, None otherwise
    """
    possible_paths = [
        'classes.txt',
        'data/classes.txt',
        '../classes.txt',
        './Industrial_Faults_Detection_Dataset/classes.txt',
        '/content/Industrial_Faults_Detection_Dataset/classes.txt'
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None


def main():
    """Main function to handle command line arguments and create YAML file."""
    parser = argparse.ArgumentParser(
        description="Generate YAML configuration file for YOLOv11 training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python create_yaml.py
    python create_yaml.py --classes_path ./classes.txt --output_path ./data.yaml
    python create_yaml.py --data_root /path/to/dataset
        """
    )
    
    parser.add_argument(
        "--classes_path",
        type=str,
        help="Path to the classes.txt file (auto-detected if not provided)"
    )
    
    parser.add_argument(
        "--output_path",
        type=str,
        default="data.yaml",
        help="Output path for the data.yaml file (default: ./data.yaml)"
    )
    
    parser.add_argument(
        "--data_root",
        type=str,
        default="/content/data",
        help="Root path for the dataset (default: /content/data for Colab)"
    )
    
    args = parser.parse_args()
    
    print("Industrial Fault Detection - YAML Configuration Generator")
    print("=" * 60)
    
    # Determine classes file path
    classes_path = args.classes_path
    if not classes_path:
        print("Searching for classes.txt file...")
        classes_path = find_classes_file()
        if not classes_path:
            print("✗ Could not find classes.txt file!")
            print("Please specify the path using --classes_path or ensure classes.txt exists in:")
            print("  - Current directory")
            print("  - data/ subdirectory")
            print("  - Dataset directory")
            sys.exit(1)
        print(f"✓ Found classes file: {classes_path}")
    
    # Create YAML configuration
    success = create_data_yaml(classes_path, args.output_path, args.data_root)
    
    if success:
        print("\n" + "=" * 60)
        print("Configuration file created successfully!")
        print("Next step: Run run_inference.py to start training")
        print(f"Data root: {args.data_root}")
        print(f"Config file: {args.output_path}")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
