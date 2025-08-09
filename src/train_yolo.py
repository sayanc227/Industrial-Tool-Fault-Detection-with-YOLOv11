# ----------------------------
# YOLOv11 Training Script
# ----------------------------

import os

# ----------------------------
# Config
# ----------------------------
DATA_YAML = "path/to/data.yaml"          # Path to your dataset config
MODEL = "yolo11s.pt"                     # Base YOLOv11 model
EPOCHS = 100
IMAGE_SIZE = 640

# ----------------------------
# Run training
# ----------------------------
os.system(f"yolo detect train data='{DATA_YAML}' model={MODEL} epochs={EPOCHS} imgsz={IMAGE_SIZE}")
