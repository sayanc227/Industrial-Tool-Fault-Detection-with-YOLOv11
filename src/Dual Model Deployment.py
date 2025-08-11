# ----------------------------
# YOLOv11 Dual Model Deployment
# ----------------------------

from ultralytics import YOLO
import cv2
import os
import glob

# ----------------------------
# Config
# ----------------------------
TOOLS_MODEL_PATH = "path/to/tools_machines_model.pt"
FAULTS_MODEL_PATH = "path/to/faults_breakdowns_model.pt"
IMAGE_FOLDER = "path/to/images"
OUTPUT_FOLDER = "path/to/output"

# ----------------------------
# Load models
# ----------------------------
tools_model = YOLO(TOOLS_MODEL_PATH)
faults_model = YOLO(FAULTS_MODEL_PATH)

# Create output folder
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Assign colors
TOOLS_COLOR = (0, 255, 0)    # Green for tools
FAULTS_COLOR = (0, 0, 255)   # Red for faults

# ----------------------------
# Process all images in folder
# ----------------------------
image_files = glob.glob(os.path.join(IMAGE_FOLDER, "*.*"))

for img_path in image_files:
    image = cv2.imread(img_path)
    if image is None:
        print(f"⚠️ Could not read {img_path}")
        continue

    # Run inference for both models
    tools_results = tools_model.predict(source=img_path, conf=0.25, verbose=False)
    faults_results = faults_model.predict(source=img_path, conf=0.25, verbose=False)

    # Draw Tools detections
    for result in tools_results[0].boxes:
        x1, y1, x2, y2 = map(int, result.xyxy[0])
        label = tools_model.names[int(result.cls[0])]
        conf = float(result.conf[0])
        cv2.rectangle(image, (x1, y1), (x2, y2), TOOLS_COLOR, 2)
        cv2.putText(image, f"{label} {conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, TOOLS_COLOR, 2)

    # Draw Faults detections
    for result in faults_results[0].boxes:
        x1, y1, x2, y2 = map(int, result.xyxy[0])
        label = faults_model.names[int(result.cls[0])]
        conf = float(result.conf[0])
        cv2.rectangle(image, (x1, y1), (x2, y2), FAULTS_COLOR, 2)
        cv2.putText(image, f"{label} {conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, FAULTS_COLOR, 2)

    # Save output image
    output_path = os.path.join(OUTPUT_FOLDER, os.path.basename(img_path))
    cv2.imwrite(output_path, image)
    print(f"✅ Saved: {output_path}")

print("🎯 All images processed and saved!")
