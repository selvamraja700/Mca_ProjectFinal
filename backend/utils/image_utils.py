import io
import base64


def pil_to_base64(pil_img, fmt='PNG'):
    """Convert PIL image to base64-encoded string."""
    buffer = io.BytesIO()
    pil_img.save(buffer, format=fmt)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')
