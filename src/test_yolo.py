# ----------------------------
# YOLOv11 Testing/Prediction Script
# ----------------------------

import os
import glob
from IPython.display import Image, display

# ----------------------------
# Config
# ----------------------------
MODEL_PATH = "path/to/best.pt"            # Trained YOLOv11 model path
SOURCE_PATH = "path/to/images"            # Folder of images for testing
SAVE_RESULTS = True

# ----------------------------
# Run prediction
# ----------------------------
os.system(f"yolo detect predict model='{MODEL_PATH}' source='{SOURCE_PATH}' save={str(SAVE_RESULTS)}")

# ----------------------------
# Display first 50 results
# ----------------------------
for image_path in glob.glob('runs/detect/predict/*.jpg')[:50]:
    display(Image(filename=image_path, height=400))

print("\n✅ Prediction complete!")
