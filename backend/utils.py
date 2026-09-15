import io
import base64
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

PAPER_METRICS = {
    "paper_title": "Digital Image Forensics: An Improved DenseNet Architecture for Forged Image Detection",
    "author": "Ahmed Alzahrani",
    "journal": "Engineering, Technology & Applied Science Research (ETASR 2024)",
    "dataset": {
        "name": "CASIA v2.0 Image Tampering Detection Database",
        "total_images": 12323,
        "authentic_count": 7200,
        "forged_count": 5123,
        "image_formats": ["JPEG", "BMP", "TIFF"],
        "resolutions": "320x240 to 800x600 color images"
    },
    "proposed_model": {
        "name": "Improved DenseNet-121",
        "accuracy": 92.32,
        "precision": 0.89,
        "recall": 0.90,
        "f1_score": 0.91,
        "total_params": 9549006,
        "trainable_params": 2511502,
        "non_trainable_params": 7037504
    },
    "confusion_matrix": {
        "true_genuine": 226,
        "false_genuine": 240,
        "true_forged": 127,
        "false_forged": 230
    },
    "comparative_results": [
        {"model": "Xception", "precision": 0.77, "recall": 0.74, "f1_score": 0.73, "accuracy": 76.51},
        {"model": "MobileNetV3-Small", "precision": 0.78, "recall": 0.75, "f1_score": 0.77, "accuracy": 72.21},
        {"model": "ResNet-50", "precision": 0.59, "recall": 0.64, "f1_score": 0.63, "accuracy": 69.24},
        {"model": "LSTM", "precision": 0.61, "recall": 0.63, "f1_score": 0.62, "accuracy": 63.31},
        {"model": "RNN", "precision": 0.71, "recall": 0.71, "f1_score": 0.71, "accuracy": 70.64},
        {"model": "Improved DenseNet (Proposed)", "precision": 0.89, "recall": 0.90, "f1_score": 0.91, "accuracy": 92.32}
    ],
    "training_epochs": [
        {"epoch": 1, "time_s": 87, "train_loss": 0.95, "train_acc": 0.65, "test_loss": 0.50, "test_acc": 0.81},
        {"epoch": 2, "time_s": 75, "train_loss": 0.38, "train_acc": 0.84, "test_loss": 0.55, "test_acc": 0.78},
        {"epoch": 3, "time_s": 75, "train_loss": 0.36, "train_acc": 0.86, "test_loss": 0.71, "test_acc": 0.72},
        {"epoch": 4, "time_s": 80, "train_loss": 0.25, "train_acc": 0.91, "test_loss": 0.41, "test_acc": 0.85},
        {"epoch": 5, "time_s": 77, "train_loss": 0.24, "train_acc": 0.89, "test_loss": 0.45, "test_acc": 0.82},
        {"epoch": 6, "time_s": 79, "train_loss": 0.20, "train_acc": 0.94, "test_loss": 0.45, "test_acc": 0.83},
        {"epoch": 7, "time_s": 100, "train_loss": 0.18, "train_acc": 0.94, "test_loss": 0.45, "test_acc": 0.84},
        {"epoch": 8, "time_s": 104, "train_loss": 0.12, "train_acc": 0.97, "test_loss": 0.50, "test_acc": 0.85},
        {"epoch": 9, "time_s": 108, "train_loss": 0.10, "train_acc": 0.97, "test_loss": 0.48, "test_acc": 0.86},
        {"epoch": 10, "time_s": 103, "train_loss": 0.11, "train_acc": 0.96, "test_loss": 0.48, "test_acc": 0.87}
    ]
}


def create_demo_authentic_image():
    img = Image.new('RGB', (400, 300), color=(135, 206, 235))
    draw = ImageDraw.Draw(img)
    draw.ellipse((300, 30, 360, 90), fill=(255, 223, 0))
    draw.polygon([(0, 300), (120, 150), (250, 300)], fill=(34, 139, 34))
    draw.polygon([(150, 300), (280, 120), (400, 300)], fill=(46, 139, 87))
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=95)
    return buf.getvalue()


def create_demo_forged_image():
    base = Image.new('RGB', (400, 300), color=(135, 206, 235))
    draw = ImageDraw.Draw(base)
    draw.ellipse((300, 30, 360, 90), fill=(255, 223, 0))
    draw.polygon([(0, 300), (120, 150), (250, 300)], fill=(34, 139, 34))
    draw.polygon([(150, 300), (280, 120), (400, 300)], fill=(46, 139, 87))
    
    buf1 = io.BytesIO()
    base.save(buf1, format='JPEG', quality=60)
    buf1.seek(0)
    base_compressed = Image.open(buf1).convert('RGB')
    
    patch = base_compressed.crop((280, 120, 360, 200))
    patch = patch.rotate(15, expand=False)
    
    base_compressed.paste(patch, (40, 50))
    
    buf2 = io.BytesIO()
    base_compressed.save(buf2, format='JPEG', quality=98)
    return buf2.getvalue()


def pil_to_base64(pil_img, fmt='PNG'):
    buffer = io.BytesIO()
    pil_img.save(buffer, format=fmt)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')
