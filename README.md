# Industrial Fault Detection with YOLOv11

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/downloads/)
[![YOLOv11](https://img.shields.io/badge/YOLOv11-Latest-green)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)


A computer vision project for detecting industrial faults using YOLOv11 object detection model. This project implements automated fault detection in industrial equipment and components using deep learning techniques.

## 🎯 Project Overview

This project applies the YOLOv11 object detection model to identify various types of industrial faults in tools and machinery. It’s designed to support automated fault detection in industrial environments by recognizing multiple defect categories. The setup includes organized datasets, dynamic configuration generation, and streamlined scripts for training and evaluation.


## 🛠️ Features

- **Automated Dataset Splitting**: Intelligent train/validation split functionality
- **Dynamic Configuration**: Automatic YAML configuration file generation
- **YOLOv11 Integration**: Utilizes the latest YOLO architecture for superior performance
- **Industrial Focus**: Specialized for industrial fault detection scenarios
- **Scalable Architecture**: Easy to extend for additional fault types

## 📁 Project Structure

```
industrial-fault-detection/
├── scripts/
│   ├── split_dataset.py          # Dataset splitting utility
│   ├── create_yaml.py             # Configuration file generator
│   ├── run_inference.py           # Model training script
│   ├── test_model.py              # Professional model testing
│   └── demo_inference.py          # Quick demo inference
├── data/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   └── validation/
│       ├── images/
│       └── labels/
├── models/
│   └── (trained model weights)
├── results/
│   └── (training results and metrics)
├── README.md
├── requirements.txt
└── data.yaml
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- Google Colab or local Python environment

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/sayanc227/industrial-fault-detection.git
   cd industrial-fault-detection
   ```

2. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your dataset**
   - Organize your images and labels according to YOLO format
   - Update the dataset path in the scripts

## 📊 Usage

### 1. Dataset Preparation

Split your dataset into training and validation sets:

```bash
python scripts/split_dataset.py --datapath="/path/to/your/dataset" --train_pct=0.9
```

### 2. Configuration Setup

Generate the YAML configuration file:

```bash
python scripts/create_yaml.py
```

This script will:
- Read your classes.txt file
- Generate data.yaml with proper paths and class information
- Configure training parameters

### 3. Model Training

Train the YOLOv11 model:

```bash
python scripts/run_inference.py
```

Training parameters:
- **Model**: YOLOv11s (small variant)
- **Epochs**: 100
- **Image Size**: 640x640
- **Format**: PyTorch (.pt)

### 4. Model Testing and Inference

After training, test your model on new images:

**Professional Testing Script:**
```bash
python scripts/test_model.py --model path/to/best.pt --source path/to/test_images/
```

**Quick Demo (matches original Colab workflow):**
```bash
python scripts/demo_inference.py
```

Testing features:
- **Automated model detection**: Finds trained models automatically
- **Batch processing**: Process entire directories of images
- **Visual results**: Display detected faults with bounding boxes
- **Confidence filtering**: Adjustable detection confidence threshold
- **Results saving**: Saves annotated images and detection labels

## 📈 Model Performance

The model achieves competitive performance in industrial fault detection:

- **Architecture**: YOLOv11s
- **Training Data Split**: 90% training, 10% validation
- **Input Resolution**: 640x640 pixels
- **Training Epochs**: 100

## 🔧 Configuration

### Dataset Structure
Ensure your dataset follows this structure:
```
dataset/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── labels/
│   ├── image1.txt
│   ├── image2.txt
│   └── ...
└── classes.txt
```

### Classes Configuration
Update `classes.txt` with your fault categories:
```
fault_type_1
fault_type_2
fault_type_3
...
```

## 💻 Development Environment

This project was developed and tested in:
- **Google Colab**: Primary development environment
- **Python**: 3.8+
- **CUDA**: For GPU acceleration
- **YOLOv11**: Latest YOLO architecture

## 📝 Scripts Description

### `split_dataset.py`
- Downloads and executes train/validation split utility
- Configurable train/test ratio (default: 90/10)
- Handles YOLO format datasets

### `create_yaml.py`
- Automatically generates YAML configuration
- Reads class names from classes.txt
- Sets up proper data paths for training
- Creates YOLOv11-compatible configuration

### `test_model.py`
- Professional model testing and inference
- Supports batch processing of images
- Configurable confidence thresholds
- Visual display of results (Jupyter/Matplotlib)
- Automatic model detection
- Saves annotated results and detection labels

### `demo_inference.py`
- Simple demo script matching original Colab workflow
- Quick testing with minimal configuration
- Automatic path detection for local/Colab environments
- Direct recreation of original inference commands

## 🎯 Applications

This industrial fault detection system can be applied to:
- **Manufacturing Quality Control**
- **Predictive Maintenance**
- **Automated Inspection Systems**
- **Industrial Safety Monitoring**
- **Equipment Health Assessment**

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Contact

**Sayan C ** - [GitHub Profile](https://github.com/sayanc227)

Project Link: [https://github.com/sayanc227/industrial-fault-detection](https://github.com/sayanc227/industrial-fault-detection)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- YOLOv11 developers for the excellent object detection framework
- EdjeElectronics for the train/validation split utility
- Google Colab for providing the development environment
- The open-source computer vision community

---

⭐ **Star this repository if you find it helpful!**
