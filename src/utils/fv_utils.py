import os
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_PATH = "../models/mediapipe_face_detector/detector.tflite"
MIN_DETECTION_CONFIDENCE = 0.5

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def list_images(root: Path):
    return [
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]


def rel_id(path: Path, root: Path) -> str:
    return str(path.relative_to(root)).replace(os.sep, "__")


def get_detector(model_path=MODEL_PATH, min_conf=MIN_DETECTION_CONFIDENCE):

    base_options = python.BaseOptions(
        model_asset_path=model_path
    )

    options = vision.FaceDetectorOptions(
        base_options=base_options,
        min_detection_confidence=min_conf,
    )

    detector = vision.FaceDetector.create_from_options(options)
    return detector

# ====================
# Metrics and Evaluation
# ====================

def create_cm(y_true, y_pred, labels=None, save_path=None, display=False):
    cm = confusion_matrix(y_true, y_pred)
    if labels:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    else:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    if save_path:
        disp.figure_.savefig(save_path)
    if display:
        plt.show()
    return cm

def create_report(y_true, y_pred, labels=None, save_path=None, display=False):
    if labels:
        report = classification_report(y_true, y_pred, target_names=labels)
    else:
        report = classification_report(y_true, y_pred)
    if save_path:
        with open(save_path, "w") as f:
            f.write(report)
    if display:
        print(report)
    return report
