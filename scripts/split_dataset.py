#!/usr/bin/env python3
"""
Dataset Splitting Script for Industrial Fault Detection

This script downloads and executes a train/validation split utility
for YOLO format datasets. It organizes images and labels into
training and validation directories.

Usage:
    python split_dataset.py --datapath="/path/to/dataset" --train_pct=0.9

Author: Sayan C 
Project: Industrial Fault Detection using YOLOv11
"""

import os
import sys
import subprocess
import argparse
import urllib.request
from pathlib import Path


def download_split_utility():
    """
    Download the train_val_split.py utility from EdjeElectronics repository.
    
    Returns:
        str: Path to the downloaded script
    """
    url = "https://raw.githubusercontent.com/EdjeElectronics/Train-and-Deploy-YOLO-Models/refs/heads/main/utils/train_val_split.py"
    script_path = "train_val_split.py"
    
    try:
        print("Downloading train/validation split utility...")
        urllib.request.urlretrieve(url, script_path)
        print(f"✓ Successfully downloaded {script_path}")
        return script_path
    except Exception as e:
        print(f"✗ Error downloading split utility: {e}")
        sys.exit(1)


def validate_dataset_path(datapath):
    """
    Validate that the dataset path exists and contains necessary files.
    
    Args:
        datapath (str): Path to the dataset directory
        
    Returns:
        bool: True if path is valid, False otherwise
    """
    if not os.path.exists(datapath):
        print(f"✗ Error: Dataset path '{datapath}' does not exist!")
        return False
    
    # Check for images and labels directories or files
    path_obj = Path(datapath)
    has_images = any(path_obj.rglob("*.jpg")) or any(path_obj.rglob("*.jpeg")) or any(path_obj.rglob("*.png"))
    
    if not has_images:
        print(f"✗ Warning: No image files found in '{datapath}'")
        print("Make sure your dataset contains .jpg, .jpeg, or .png files")
    
    return True


def split_dataset(datapath, train_pct=0.9):
    """
    Execute the dataset splitting process.
    
    Args:
        datapath (str): Path to the dataset directory
        train_pct (float): Percentage of data to use for training (0.0 to 1.0)
    """
    # Validate inputs
    if not 0.0 < train_pct < 1.0:
        print("✗ Error: train_pct must be between 0.0 and 1.0")
        sys.exit(1)
    
    if not validate_dataset_path(datapath):
        sys.exit(1)
    
    # Download the split utility
    script_path = download_split_utility()
    
    try:
        # Execute the splitting script
        print(f"\nSplitting dataset...")
        print(f"Dataset path: {datapath}")
        print(f"Training percentage: {train_pct * 100}%")
        print(f"Validation percentage: {(1 - train_pct) * 100}%")
        print("-" * 50)
        
        cmd = [
            sys.executable, script_path,
            f"--datapath={datapath}",
            f"--train_pct={train_pct}"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ Dataset split completed successfully!")
            print("\nOutput:")
            print(result.stdout)
        else:
            print("✗ Error during dataset splitting:")
            print(result.stderr)
            sys.exit(1)
            
    except Exception as e:
        print(f"✗ Error executing split script: {e}")
        sys.exit(1)
    
    finally:
        # Clean up downloaded script
        if os.path.exists(script_path):
            os.remove(script_path)
            print(f"Cleaned up temporary file: {script_path}")


def main():
    """Main function to handle command line arguments and execute splitting."""
    parser = argparse.ArgumentParser(
        description="Split dataset into training and validation sets for YOLO training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python split_dataset.py --datapath="/path/to/dataset" --train_pct=0.9
    python split_dataset.py --datapath="./Industrial_Faults_Dataset" --train_pct=0.8
        """
    )
    
    parser.add_argument(
        "--datapath",
        type=str,
        required=True,
        help="Path to the dataset directory containing images and labels"
    )
    
    parser.add_argument(
        "--train_pct",
        type=float,
        default=0.9,
        help="Percentage of data to use for training (default: 0.9)"
    )
    
    args = parser.parse_args()
    
    print("Industrial Fault Detection - Dataset Splitting")
    print("=" * 50)
    
    # Execute dataset splitting
    split_dataset(args.datapath, args.train_pct)
    
    print("\n" + "=" * 50)
    print("Dataset preparation completed!")
    print("Next step: Run create_yaml.py to generate configuration file")


if __name__ == "__main__":
    main()
