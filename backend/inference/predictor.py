import numpy as np
import torch

from preprocessing.image_loader import preprocess_image_bytes
from preprocessing.ela import compute_ela, generate_heatmap_overlay
from utils.image_utils import pil_to_base64


def predict_image(model, image_bytes, filename="uploaded_image.jpg"):
    """
    Run full digital forgery prediction pipeline on input image bytes.
    Orchestrates: preprocessing → inference → ELA → heatmap → result.
    """
    pil_img, input_tensor = preprocess_image_bytes(image_bytes)

    with torch.no_grad():
        logits, probabilities = model(input_tensor)
        prob_vec = probabilities[0].tolist()

    prob_real = float(prob_vec[0])
    prob_fake = float(prob_vec[1])

    ela_img, ela_b64 = compute_ela(pil_img)
    heatmap_b64 = generate_heatmap_overlay(pil_img, ela_img)

    ela_noise = float(np.array(ela_img, dtype=np.float32).mean())

    is_forged = prob_fake > 0.5 or (prob_fake > 0.40 and ela_noise > 20.0)
    confidence = prob_fake if is_forged else prob_real

    prediction_class = "Forged" if is_forged else "Genuine"
    vector_label = [0, 1] if is_forged else [1, 0]

    thumb = pil_img.copy()
    thumb.thumbnail((400, 400))
    orig_b64 = pil_to_base64(thumb)

    return {
        "success": True,
        "filename": filename,
        "prediction": prediction_class,
        "vector": vector_label,
        "probabilities": {
            "genuine": round(prob_real * 100, 2),
            "forged": round(prob_fake * 100, 2)
        },
        "confidence": round(confidence * 100, 2),
        "ela_noise_level": round(ela_noise, 2),
        "dimensions": f"{pil_img.width}x{pil_img.height}",
        "visualizations": {
            "original": f"data:image/png;base64,{orig_b64}",
            "ela": f"data:image/png;base64,{ela_b64}",
            "heatmap": f"data:image/png;base64,{heatmap_b64}"
        }
    }
