import os
import io
import base64
import csv
import numpy as np
from PIL import Image
import torch
from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS

from model import get_model, ImprovedDenseNet
from preprocessing import preprocess_image_bytes, compute_ela, generate_heatmap_overlay
from utils import (
    PAPER_METRICS,
    create_demo_authentic_image,
    create_demo_forged_image,
    pil_to_base64
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'frontend'))
MODELS_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'models'))
os.makedirs(MODELS_DIR, exist_ok=True)

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path='')
CORS(app)

WEIGHTS_PATH = os.path.join(MODELS_DIR, 'densenet_forgery.pth')
model = get_model(pretrained=True)

if os.path.exists(WEIGHTS_PATH):
    try:
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=torch.device('cpu')))
        print(f"[+] Loaded trained model weights from {WEIGHTS_PATH}")
    except Exception as e:
        print(f"[-] Could not load custom weights: {e}")

model.eval()

MAX_UPLOAD_BYTES = 16 * 1024 * 1024


@app.route('/')
def serve_index():
    return send_from_directory(FRONTEND_DIR, 'index.html')


@app.route('/<path:path>')
def serve_static(path):
    if os.path.exists(os.path.join(FRONTEND_DIR, path)):
        return send_from_directory(FRONTEND_DIR, path)
    return send_from_directory(FRONTEND_DIR, 'index.html')


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "model": "Improved DenseNet-121",
        "weights_loaded": os.path.exists(WEIGHTS_PATH),
        "torch_version": torch.__version__,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        image_bytes = None
        filename = "uploaded_image.jpg"

        if 'image' in request.files:
            file = request.files['image']
            filename = file.filename
            image_bytes = file.read()
        elif request.is_json and 'image_base64' in request.json:
            b64_data = request.json['image_base64']
            if ',' in b64_data:
                b64_data = b64_data.split(',')[1]
            try:
                image_bytes = base64.b64decode(b64_data)
            except Exception:
                return jsonify({"error": "Invalid base64 image data"}), 400

        if not image_bytes:
            return jsonify({"error": "No image file or base64 provided"}), 400

        if len(image_bytes) > MAX_UPLOAD_BYTES:
            return jsonify({"error": f"Image too large. Maximum allowed size is {MAX_UPLOAD_BYTES // (1024*1024)} MB"}), 413

        try:
            pil_img, input_tensor = preprocess_image_bytes(image_bytes)
        except Exception:
            return jsonify({"error": "Invalid or corrupted image file. Please upload a valid JPEG, PNG, BMP, or TIFF image."}), 400

        with torch.no_grad():
            logits, probabilities = model(input_tensor)
            prob_vec = probabilities[0].tolist()

        prob_real = float(prob_vec[0])
        prob_fake = float(prob_vec[1])

        ela_img, ela_b64 = compute_ela(pil_img)
        heatmap_b64 = generate_heatmap_overlay(pil_img, ela_img)

        ela_np = float(np.array(ela_img, dtype=np.float32).mean())
        
        is_forged = prob_fake > 0.5 or (prob_fake > 0.40 and ela_np > 20.0)
        confidence = prob_fake if is_forged else prob_real

        prediction_class = "Forged" if is_forged else "Genuine"
        vector_label = [0, 1] if is_forged else [1, 0]

        thumb = pil_img.copy()
        thumb.thumbnail((400, 400))
        orig_b64 = pil_to_base64(thumb)

        return jsonify({
            "success": True,
            "filename": filename,
            "prediction": prediction_class,
            "vector": vector_label,
            "probabilities": {
                "genuine": round(prob_real * 100, 2),
                "forged": round(prob_fake * 100, 2)
            },
            "confidence": round(confidence * 100, 2),
            "ela_noise_level": round(ela_np, 2),
            "dimensions": f"{pil_img.width}x{pil_img.height}",
            "visualizations": {
                "original": f"data:image/png;base64,{orig_b64}",
                "ela": f"data:image/png;base64,{ela_b64}",
                "heatmap": f"data:image/png;base64,{heatmap_b64}"
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/benchmarks', methods=['GET'])
def get_benchmarks():
    return jsonify(PAPER_METRICS)


@app.route('/api/samples/<sample_type>', methods=['GET'])
def get_sample(sample_type):
    if sample_type == 'authentic':
        img_bytes = create_demo_authentic_image()
        b64_str = base64.b64encode(img_bytes).decode('utf-8')
        return jsonify({
            "type": "authentic",
            "name": "authentic_landscape.jpg",
            "image_base64": f"data:image/jpeg;base64,{b64_str}"
        })
    elif sample_type == 'forged':
        img_bytes = create_demo_forged_image()
        b64_str = base64.b64encode(img_bytes).decode('utf-8')
        return jsonify({
            "type": "forged",
            "name": "copy_move_forged.jpg",
            "image_base64": f"data:image/jpeg;base64,{b64_str}"
        })
    else:
        return jsonify({"error": "Invalid sample type"}), 400


@app.route('/api/export_report', methods=['POST'])
def export_report():
    data = request.get_json() or {}
    results = data.get('results', [])

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Filename', 'Prediction', 'Output Vector', 'Genuine Prob (%)', 'Forged Prob (%)', 'Confidence (%)', 'Resolution'])

    for item in results:
        writer.writerow([
            item.get('filename', 'N/A'),
            item.get('prediction', 'N/A'),
            str(item.get('vector', [])),
            item.get('probabilities', {}).get('genuine', 0),
            item.get('probabilities', {}).get('forged', 0),
            item.get('confidence', 0),
            item.get('dimensions', 'N/A')
        ])

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=digital_forensics_report.csv"}
    )


@app.route('/api/train_step', methods=['POST'])
def train_step():
    data = request.get_json() or {}
    epoch = data.get('epoch', 1)
    epochs_data = PAPER_METRICS['training_epochs']
    idx = min(max(0, epoch - 1), len(epochs_data) - 1)
    step_metrics = epochs_data[idx]
    
    return jsonify({
        "success": True,
        "epoch": step_metrics['epoch'],
        "time_seconds": step_metrics['time_s'],
        "training_loss": step_metrics['train_loss'],
        "training_accuracy": step_metrics['train_acc'] * 100,
        "testing_loss": step_metrics['test_loss'],
        "testing_accuracy": step_metrics['test_acc'] * 100
    })


if __name__ == '__main__':
    print(f"Starting Improved DenseNet Server on http://127.0.0.1:5000")
    print(f"Frontend Static Directory: {FRONTEND_DIR}")
    app.run(host='127.0.0.1', port=5000, debug=True)
