# Industrial Object & Fault Detection Pipeline 🏭

![Python](https://img.shields.io/badge/Python-3.9%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange) ![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)

A comprehensive computer vision project that uses a two-stage YOLOv8 pipeline to first identify industrial tools and machinery, and then inspect them for common faults and breakdowns. This project demonstrates an end-to-end workflow from data collection and annotation to training, evaluation, and pipeline deployment.



---
## 🎯 Project Overview

In industrial environments, the ability to quickly and accurately identify equipment and diagnose potential issues is critical for safety and operational efficiency. This project automates this process using a two-model computer vision pipeline:

1.  **Object Detection Model:** A YOLOv8 model trained on **88 classes** of common industrial tools and machinery. Its job is to answer the question: *"What is this object?"*
2.  **Fault Detection Model:** A second, specialized YOLOv8 model trained to detect common visual defects like `rust`, `leaks`, `cracks`, and `burn marks`. Its job is to answer the question: *"Is there a problem with this object?"*

---
## ✨ Key Features

* **Two-Model Pipeline:** Utilizes a sequential inference process for high-accuracy, context-aware fault detection.
* **Custom Datasets:** Built from scratch with two personally annotated datasets for specific industrial use cases.
* **High-Performance Models:** Leverages the state-of-the-art YOLOv8 architecture for fast and accurate detections.
* **Domain-Specific:** Tailored for industrial maintenance, providing a practical solution for real-world challenges.

---
## ⚙️ How It Works

The system uses a two-pass approach to analyze an image, mimicking the workflow of a human maintenance expert.

1.  **Pass 1: Find the Equipment**
    * The main **Object Detection Model** scans the full image to locate and identify all known tools and machinery.

2.  **Pass 2: Inspect for Faults**
    * For each piece of equipment found in Pass 1, its bounding box is cropped from the main image.
    * This smaller, cropped image is then fed into the specialized **Fault Detection Model**.
    * The fault model's only task is to find defects *within* that specific piece of equipment, which greatly improves accuracy and reduces false positives.

3.  **Final Output**
    * The results are combined to produce a final, annotated image showing both the identified equipment (e.g., in a green box) and any detected faults (e.g., in a red box).

---
## 🚀 Getting Started

### Prerequisites

* Python 3.9+
* PyTorch
* An NVIDIA GPU with CUDA (recommended for performance)

### Installation

1.  Clone the repository:
    ```bash
    git clone [Your Repository URL]
    cd [Your Repository Name]
    ```

2.  Install the required Python packages:
    ```bash
    pip install ultralytics matplotlib opencv-python
    ```

### Usage

1.  Place your trained model weights (`object_model_best.pt` and `fault_model_best.pt`) in a `weights/` directory.
2.  Place the images you want to test in an `input_images/` directory.
3.  Run the inference pipeline script:
    ```bash
    python run_pipeline.py --image path/to/your/image.jpg
    ```

---
## 📊 Showcase & Demo

Here is a demonstration of the pipeline analyzing an image of an industrial pump and correctly identifying both the pump and a leak.

*You should replace this text with a GIF or a high-quality screenshot of your model in action. A GIF is highly recommended!*

**[Link to a GIF of your project in action]**

---
## 🔮 Future Improvements

* **Deployment as a Web App:** Package the pipeline into a Flask or FastAPI web application for easy use.
* **Real-Time Video Processing:** Adapt the script to run on a live video feed from an IP camera for continuous monitoring.
* **Expand Fault Classes:** Collect more data for rarer and more subtle defect types to further improve the model's diagnostic capabilities.




## 📦 Dataset
This project uses two custom datasets:

1. **Tools & Machines Detection Dataset**  
   - Annotated for object detection with YOLOv11 format  
   - [Download from Google Drive](https://drive.google.com/drive/folders/1ch4IvZ2BCWRgodM2Z0rULLT2fAn0QYK0?usp=sharing)

2. **Faults & Breakdowns Detection Dataset**  
   - Annotated for object detection with YOLOv11 format  
   - [Download from Google Drive](https://drive.google.com/drive/folders/1L0vN75vwAAZgR7LKYCC4mCztMuCCWajL?usp=sharing)

