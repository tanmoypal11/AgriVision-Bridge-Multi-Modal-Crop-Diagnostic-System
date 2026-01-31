import torch
import numpy as np
import cv2

# Load YOLOv5 model directly from torch hub
@torch.no_grad()
def load_detection_model():
    model = torch.hub.load(
        "ultralytics/yolov5",
        "yolov5s",
        pretrained=True
    )
    model.eval()
    return model


def detect_objects(image):
    """
    image: PIL Image or numpy array
    returns: list of detections
    """
    model = load_detection_model()

    # Run inference
    results = model(image)

    detections = []
    df = results.pandas().xyxy[0]

    for _, row in df.iterrows():
        detections.append({
            "label": row["name"],
            "confidence": float(row["confidence"]),
            "bbox": [
                int(row["xmin"]),
                int(row["ymin"]),
                int(row["xmax"]),
                int(row["ymax"]),
            ]
        })

    return detections


def draw_boxes(image, detections):
    """
    image: numpy array (BGR)
    """
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = f"{det['label']} {det['confidence']:.2f}"

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    return image
