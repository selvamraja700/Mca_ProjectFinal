import io
import base64
import numpy as np
from PIL import Image, ImageEnhance, ImageChops
import torch
import torchvision.transforms as transforms


IMAGE_SIZE = (224, 224)

transform_pipeline = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
])


def preprocess_image_bytes(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    tensor = transform_pipeline(img).unsqueeze(0)
    return img, tensor


def compute_ela(image, quality=90, scale=15):
    img_rgb = image.convert('RGB')
    
    buffer = io.BytesIO()
    img_rgb.save(buffer, 'JPEG', quality=quality)
    buffer.seek(0)
    
    recompressed_img = Image.open(buffer)
    
    ela_img = ImageChops.difference(img_rgb, recompressed_img)
    
    extrema = ela_img.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    
    scale_factor = scale * (255.0 / max_diff)
    ela_img = ImageEnhance.Brightness(ela_img).enhance(scale_factor)
    
    ela_buffer = io.BytesIO()
    ela_img.save(ela_buffer, format='PNG')
    ela_base64 = base64.b64encode(ela_buffer.getvalue()).decode('utf-8')
    
    return ela_img, ela_base64


def generate_heatmap_overlay(original_img, ela_img):
    ela_resized = ela_img.resize(original_img.size)
    ela_np = np.array(ela_resized, dtype=np.float32)
    
    gray_ela = np.mean(ela_np, axis=2)
    norm_ela = (gray_ela - np.min(gray_ela)) / (np.max(gray_ela) - np.min(gray_ela) + 1e-5)
    
    heatmap_np = np.zeros((norm_ela.shape[0], norm_ela.shape[1], 3), dtype=np.uint8)
    heatmap_np[:, :, 0] = (norm_ela * 255).astype(np.uint8)
    heatmap_np[:, :, 2] = ((1 - norm_ela) * 255).astype(np.uint8)
    
    heatmap_pil = Image.fromarray(heatmap_np)
    
    blended = Image.blend(original_img.convert('RGB'), heatmap_pil, alpha=0.4)
    
    buffer = io.BytesIO()
    blended.save(buffer, format='PNG')
    heatmap_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return heatmap_base64
