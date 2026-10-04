import os
from PIL import Image, ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


class CasiaDataset(Dataset):
    """
    PyTorch Dataset for CASIA v2.0 image tampering detection.
    Scans directory recursively:
      - Au/au/authentic/real/genuine  → Label 0 (Authentic)
      - Tp/tp/tampered/forged/fake/splice/cm → Label 1 (Tampered)
    """
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []

        for root, dirs, files in os.walk(data_dir):
            folder_name = os.path.basename(root)
            for fname in files:
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
                    full_path = os.path.join(root, fname)
                    if folder_name in ('Au', 'au', 'authentic', 'real', 'genuine'):
                        self.samples.append((full_path, 0))
                    elif folder_name in ('Tp', 'tp', 'tampered', 'forged', 'fake', 'splice', 'cm'):
                        self.samples.append((full_path, 1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        if self.transform:
            image = self.transform(image)
        return image, label
