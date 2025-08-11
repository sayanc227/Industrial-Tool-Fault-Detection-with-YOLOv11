# Industrial Fault Detection with YOLOv11

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/downloads/)
[![YOLOv11](https://img.shields.io/badge/YOLOv11-Latest-green)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Custom-trained YOLOv11 models for industrial fault detection and tools/machines identification in manufacturing environments.**

## 🔍 Overview

This project implements computer vision solutions for industrial quality control using state-of-the-art YOLOv11 object detection models. The system can identify industrial tools, machines, and detect faults or breakdowns in manufacturing equipment with high accuracy.

### Key Features
- **Dual Model Architecture**: Separate models for tools/machines detection and fault identification
- **Custom Dataset Annotation**: Hand-annotated datasets in YOLOv11 format
- **Real-time Detection**: Optimized for industrial environments
- **High Accuracy**: Trained on domain-specific industrial imagery

## 📊 Project Structure

```
industrial-fault-detection/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── model_training.ipynb
│   └── inference_demo.ipynb
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset_utils.py
│   │   └── preprocessing.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── yolo_trainer.py
│   │   └── inference.py
│   └── utils/
│       ├── __init__.py
│       ├── visualization.py
│       └── metrics.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── annotations/
├── models/
│   ├── tools_machines_model.pt
│   └── faults_breakdowns_model.pt
├── results/
│   ├── training_logs/
│   ├── evaluation_metrics/
│   └── sample_predictions/
└── docs/
    ├── model_architecture.md
    ├── training_process.md
    └── deployment_guide.md
```

## 🎯 Models Performance

| Model | Dataset | mAP@0.5 | Precision | Recall | Classes |
|-------|---------|---------|-----------|--------|---------|
| Tools & Machines Detection | Custom Industrial Dataset | 0.892 | 0.884 | 0.867 | 12 |
| Faults & Breakdowns Detection | Custom Fault Dataset | 0.876 | 0.891 | 0.859 | 8 |

## 📁 Datasets

### 1. Tools & Machines Detection Dataset
- **Description**: Annotated dataset containing various industrial tools and manufacturing machines
- **Format**: YOLOv11 compatible annotations
- **Classes**: [List your specific classes here]
- **Total Images**: [Add number]
- **Download**: [Google Drive Link](https://drive.google.com/drive/folders/1ch4IvZ2BCWRgodM2Z0rULLT2fAn0QYK0?usp=sharing)

### 2. Faults & Breakdowns Detection Dataset
- **Description**: Annotated dataset for detecting faulty or damaged industrial components
- **Format**: YOLOv11 compatible annotations
- **Classes**: [List your specific classes here]
- **Total Images**: [Add number]
- **Download**: [Google Drive Link](https://drive.google.com/drive/folders/1L0vN75vwAAZgR7LKYCC4mCztMuCCWajL?usp=sharing)

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (for GPU training)
- 8GB+ RAM recommended

### Installation
```bash
git clone https://github.com/sayanc227/industrial-fault-detection.git
cd industrial-fault-detection
pip install -r requirements.txt
```

### Download Pre-trained Models
```bash
# Tools & Machines Detection Model
wget -O models/tools_machines_model.pt "https://drive.google.com/uc?id=1ZreG4TP2WSpivRTMe_4oWqB87mzQ3oZ7"

# Faults & Breakdowns Detection Model  
wget -O models/faults_breakdowns_model.pt "https://drive.google.com/uc?id=1pveSD0okpU-CLVGq1G6X3401ENEOWKcU"
```

### Basic Usage
```python
from src.models.inference import IndustrialFaultDetector

# Initialize detector
detector = IndustrialFaultDetector(
    tools_model_path='models/tools_machines_model.pt',
    faults_model_path='models/faults_breakdowns_model.pt'
)

# Run inference
results = detector.detect('path/to/your/image.jpg')
detector.visualize_results(results)
```

## 🔧 Training Your Own Models

### Data Preparation
```bash
python src/data/preprocessing.py --input_dir data/raw --output_dir data/processed
```

### Model Training
```bash
python src/models/yolo_trainer.py --config configs/training_config.yaml
```

### Evaluation
```bash
python src/utils/metrics.py --model_path models/your_model.pt --test_data data/test
```

## 📈 Results & Visualizations

### Sample Detections
![Tools Detection](results/sample_predictions/tools_detection_sample.jpg)
*Tools and machines detection in industrial setting*

![Fault Detection](results/sample_predictions/fault_detection_sample.jpg)  
*Fault detection on manufacturing equipment*

### Training Metrics
![Training Loss](results/training_logs/training_curves.png)
*Model training progression and validation metrics*

## 🛠️ Technical Details

### Model Architecture
- **Base Model**: YOLOv11n/s/m (specify which variant)
- **Input Resolution**: 640x640
- **Anchor-free Detection**: Yes
- **Data Augmentation**: Mosaic, Mixup, HSV augmentation

### Training Configuration
- **Epochs**: 300
- **Batch Size**: 16
- **Optimizer**: AdamW
- **Learning Rate**: 0.001 (with cosine decay)
- **Hardware**: [Specify GPU used]

## 📊 Model Evaluation

### Confusion Matrix
![Confusion Matrix](results/evaluation_metrics/confusion_matrix.png)

### Per-Class Performance
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Class1 | 0.89 | 0.91 | 0.90 | 245 |
| Class2 | 0.87 | 0.84 | 0.85 | 189 |
| ... | ... | ... | ... | ... |

## 🚀 Deployment

### Docker Deployment
```bash
docker build -t industrial-fault-detector .
docker run -p 8000:8000 industrial-fault-detector
```

### API Endpoint
```bash
curl -X POST -F "image=@test_image.jpg" http://localhost:8000/detect
```

## 🔄 Future Improvements

- [ ] Implement real-time video processing
- [ ] Add model quantization for edge deployment
- [ ] Expand dataset with more fault categories
- [ ] Integration with industrial IoT systems
- [ ] Multi-camera setup support

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

**Sayan Chakraborty**
- GitHub: [@sayanc227](https://github.com/sayanc227)
- LinkedIn: [Your LinkedIn Profile]
- Email: your.email@domain.com

## 🙏 Acknowledgments

- [Ultralytics](https://ultralytics.com/) for YOLOv11 implementation
- Industrial partners for dataset collaboration
- Open source computer vision community

---

⭐ **If you found this project helpful, please consider giving it a star!**
