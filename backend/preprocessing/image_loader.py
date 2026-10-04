import io
from PIL import Image
import torchvision.transforms as transforms
from config.settings import IMAGE_SIZE

transform_pipeline = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
])


def preprocess_image_bytes(image_bytes):
    """Load image bytes into PIL Image (RGB) and transformed PyTorch tensor (batch dim added)."""
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    tensor = transform_pipeline(img).unsqueeze(0)
    return img, tensor
