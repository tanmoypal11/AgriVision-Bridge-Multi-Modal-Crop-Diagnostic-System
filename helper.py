import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from datetime import datetime
import os

# -----------------------------
# Load YOLO model
# -----------------------------
def load_model(weights_path: str):
    """
    Load YOLO model from weights
    """
    return YOLO(weights_path)


# -----------------------------
# Run inference
# -----------------------------
def run_detection(model, image):
    """
    Run YOLO inference on image
    Returns YOLO results
    """
    results = model(image)
    return results


# -----------------------------
# Draw bounding boxes
# -----------------------------
def draw_boxes(image, results):
    """
    Draw bounding boxes and labels on image
    """
    annotated_img = image.copy()

    for r in results:
        boxes = r.boxes
        names = r.names

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = f"{names[cls]} {conf:.2f}"

            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated_img,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

    return annotated_img


# -----------------------------
# Extract predictions for table / LLM
# -----------------------------
def extract_predictions(results):
    """
    Extract prediction details in list format
    """
    predictions = []

    for r in results:
        names = r.names
        for box in r.boxes:
            predictions.append({
                "class": names[int(box.cls[0])],
                "confidence": round(float(box.conf[0]), 3),
                "x1": int(box.xyxy[0][0]),
                "y1": int(box.xyxy[0][1]),
                "x2": int(box.xyxy[0][2]),
                "y2": int(box.xyxy[0][3]),
            })

    return predictions


# -----------------------------
# Save results to CSV
# -----------------------------
def save_results_to_csv(predictions, csv_path="results.csv"):
    """
    Append detection results to CSV
    """
    if not predictions:
        return

    df = pd.DataFrame(predictions)
    df["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)


# -----------------------------
# Convert uploaded image to OpenCV
# -----------------------------
def read_image(uploaded_file):
    """
    Convert Streamlit uploaded file to OpenCV image
    """
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image
