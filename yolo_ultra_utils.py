
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image
import torch

# Load YOLOv5s model
yolo_model = YOLO('yolov5s.pt')  # Automatically downloads weights

# Detection function
def detect_objects_yolo_ultralytics(img_tensor, conf_thresh=0.5):
    # Convert tensor to PIL image
    pil_img = transforms.ToPILImage()(img_tensor.cpu()).convert("RGB")

    # Run prediction
    results = yolo_model.predict(pil_img, conf=conf_thresh, verbose=False)

    # Extract labels
    labels = []
    for r in results:
        for box in r.boxes:
            label = yolo_model.names[int(box.cls)]
            labels.append(label)

    return list(set(labels))
