
import cv2
import torch
from ultralytics import YOLO
from pathlib import Path
import logging

# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_pipeline(object_model_path, fault_model_path, image_path, object_conf=0.5, fault_conf=0.3):
    """
    Runs a two-pass object and fault detection pipeline on an image.

    Args:
        object_model_path (str): Path to the object detection model weights.
        fault_model_path (str): Path to the fault detection model weights.
        image_path (str): Path to the input image.
        object_conf (float): Confidence threshold for the object model.
        fault_conf (float): Confidence threshold for the fault model.

    Returns:
        numpy.ndarray: The image with combined detections drawn on it, or None if an error occurs.
    """
    # --- Load Models ---
    try:
        logging.info("Loading models...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        object_model = YOLO(object_model_path)
        fault_model = YOLO(fault_model_path)
        logging.info("✅ Models loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load models. Error: {e}")
        return None

    # --- Load Image ---
    if not Path(image_path).exists():
        logging.error(f"Image not found at {image_path}")
        return None

    full_image = cv2.imread(image_path)
    if full_image is None:
        logging.error(f"Failed to read image at {image_path}")
        return None

    final_image = full_image.copy()

    # --- 1. First Pass: Detect Objects (Tools & Machinery) ---
    logging.info("Running Pass 1: Detecting Tools & Machinery...")
    object_results = object_model.predict(source=full_image, conf=object_conf, device=device, verbose=False)

    # Loop through each detected object
    for result in object_results:
        for box in result.boxes:
            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Get class name and confidence
            obj_class_id = int(box.cls[0])
            obj_class_name = object_model.names[obj_class_id]
            obj_confidence = float(box.conf[0])

            # Draw the box for the object on the final image (optional, can be moved to visualization)
            cv2.rectangle(final_image, (x1, y1), (x2, y2), (0, 255, 0), 2) # Green for objects
            cv2.putText(final_image, f"{obj_class_name} ({obj_confidence:.2f})",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # --- 2. Second Pass: Detect Faults within the Object's BBox ---
            logging.info(f"  -> Cropping '{obj_class_name}' and running Pass 2 for faults...")

            # Ensure crop coordinates are within image bounds
            y1_crop = max(0, y1)
            y2_crop = min(full_image.shape[0], y2)
            x1_crop = max(0, x1)
            x2_crop = min(full_image.shape[1], x2)

            cropped_image = full_image[y1_crop:y2_crop, x1_crop:x2_crop]

            if cropped_image.shape[0] > 0 and cropped_image.shape[1] > 0: # Check if crop is valid
                # Run the fault model on the cropped image
                fault_results = fault_model.predict(source=cropped_image, conf=fault_conf, device=device, verbose=False)

                for fault_result in fault_results:
                    if len(fault_result.boxes) > 0:
                        logging.info(f"    🚨 FAULTS DETECTED for '{obj_class_name}'!")
                    for fault_box in fault_result.boxes:
                        # Get fault coordinates relative to the CROPPED image
                        fx1_crop, fy1_crop, fx2_crop, fy2_crop = map(int, fault_box.xyxy[0])

                        # Convert coordinates back to the FULL image space
                        fx1_full, fy1_full = x1_crop + fx1_crop, y1_crop + fy1_crop
                        fx2_full, fy2_full = x1_crop + fx2_crop, y1_crop + fy2_crop

                        # Get fault class name and confidence
                        fault_class_id = int(fault_box.cls[0])
                        fault_class_name = fault_model.names[fault_class_id]
                        fault_confidence = float(fault_box.conf[0])

                        # Draw the box for the fault on the final image (optional, can be moved to visualization)
                        cv2.rectangle(final_image, (fx1_full, fy1_full), (fx2_full, fy2_full), (0, 0, 255), 2) # Red for faults
                        cv2.putText(final_image, f"  - {fault_class_name} ({fault_confidence:.2f})",
                                    (fx1_full, fy1_full - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                logging.warning(f"  -> Skipping fault detection for empty crop area of '{obj_class_name}'")


    logging.info("✅ Pipeline execution complete.")
    return final_image

# Example usage (will be removed when used as part of a package)
# if __name__ == "__main__":
#     # Define paths (example paths, replace with your actual paths)
#     OBJECT_MODEL_PATH = "/content/best.pt"
#     FAULT_MODEL_PATH = "/content/sample_data/best.pt"
#     IMAGE_PATH = "/content/drive/MyDrive/YOLO/custom_data/images/Screenshot_10-6-2025_114613_cfrouting.zoeysite.com.jpeg"

#     processed_image = run_pipeline(OBJECT_MODEL_PATH, FAULT_MODEL_PATH, IMAGE_PATH)

#     if processed_image is not None:
#         # Display the final image using matplotlib (assuming matplotlib is available)
#         import matplotlib.pyplot as plt
#         processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
#         plt.figure(figsize=(15, 15))
#         plt.imshow(processed_image_rgb)
#         plt.title("Combined Object and Fault Detections")
#         plt.axis('off')
#         plt.show()

# # Note: The simple inference loop from cell 6OsRLQ1HTw01 could also be a separate function in this file
# # and the display logic from cell xz9Vp-Gav8s0 would go into visualization.py
