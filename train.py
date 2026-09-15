import os
import argparse
import time

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
except ImportError:
    print("=" * 65)
    print("  [!] PyTorch ('torch' / 'torchvision') is not installed in this Python environment.")
    print("=" * 65)
    print("\nTo fix this issue, please run one of the following commands:")
    print("\nOption 1: If using a Virtual Environment (conda/venv), activate it first:")
    print("    .\\venv\\Scripts\\activate   (or your conda activate environment)")
    print("\nOption 2: Install PyTorch and required dependencies in your active Python environment:")
    print("    pip install torch torchvision pillow numpy")
    print("    pip install -r backend/requirements.txt")
    print("=" * 65)
    import sys
    sys.exit(1)

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))
from model import ImprovedDenseNet


class CasiaDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        au_count = 0
        tp_count = 0

        for root, dirs, files in os.walk(data_dir):
            folder_name = os.path.basename(root)
            for fname in files:
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
                    full_path = os.path.join(root, fname)
                    if folder_name in ('Au', 'au', 'authentic', 'real', 'genuine'):
                        self.samples.append((full_path, 0))
                        au_count += 1
                    elif folder_name in ('Tp', 'tp', 'tampered', 'forged', 'fake', 'splice', 'cm'):
                        self.samples.append((full_path, 1))
                        tp_count += 1

        print(f"[*] Dataset loaded: {au_count} Authentic (label=0) + {tp_count} Tampered (label=1) = {au_count + tp_count} Total images")

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


def train_model(data_dir, epochs=10, batch_size=19, lr=0.0001, save_path='models/densenet_forgery.pth', resume=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Training Improved DenseNet-121 on device: {device}")

    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    dataset = CasiaDataset(data_dir, transform=data_transform)
    if len(dataset) == 0:
        print(f"[!] Warning: No images found in {data_dir}. Ensure structure contains 'au/' and 'tp/' subfolders.")
        return

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = ImprovedDenseNet(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    best_acc = 0.0
    start_epoch = 1

    checkpoint_path = save_path.replace('.pth', '_checkpoint.pth')
    if resume and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        print(f"[+] Resuming from epoch {start_epoch} (best acc so far: {best_acc:.2f}%)")
    elif resume:
        print(f"[!] No checkpoint found at {checkpoint_path}, starting from scratch.")

    print(f"[*] Training on {train_size} images, testing on {test_size} images for {epochs} epochs...")
    print("=" * 65)

    for epoch in range(start_epoch, epochs + 1):
        start_t = time.time()
        model.train()
        running_loss, running_corrects = 0.0, 0

        if device.type == 'cuda':
            torch.cuda.empty_cache()

        try:
            for step, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                logits, _ = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                _, preds = torch.max(logits, 1)
                batch_loss = loss.item()
                batch_corrects = torch.sum(preds == labels.data).item()
                batch_size_actual = images.size(0)
                running_loss += batch_loss * batch_size_actual
                running_corrects += batch_corrects

                del images, labels, logits, loss, preds
                if device.type == 'cuda' and (step + 1) % 100 == 0:
                    torch.cuda.empty_cache()

                if (step + 1) % 50 == 0 or (step + 1) == len(train_loader):
                    batch_acc = (batch_corrects / batch_size_actual) * 100
                    print(f"  --> Batch {step + 1}/{len(train_loader)} - Loss: {batch_loss:.4f} - Batch Acc: {batch_acc:.1f}%", flush=True)
        except RuntimeError as e:
            import traceback
            print(f"\n[!] ERROR during training at Epoch {epoch}, Batch {step+1}:")
            traceback.print_exc()
            print("\nSkipping rest of this epoch and continuing...")
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            continue

        epoch_loss = running_loss / train_size
        epoch_acc = (running_corrects / train_size) * 100

        model.eval()
        test_corrects = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                logits, _ = model(images)
                _, preds = torch.max(logits, 1)
                test_corrects += torch.sum(preds == labels.data)

        test_acc = (test_corrects.item() / test_size) * 100
        elapsed = time.time() - start_t

        print(f"Epoch {epoch:2d}/{epochs} [{elapsed:.0f}s] - Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}% | Test Acc: {test_acc:.2f}%")

        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'best_acc': best_acc
        }, checkpoint_path)

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), save_path)
            print(f"   --> Saved best model to {save_path} (Test Acc: {best_acc:.2f}%)")

    print("=" * 65)
    print(f"[*] Training Complete! Best Test Accuracy: {best_acc:.2f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Improved DenseNet-121 for Image Forgery Detection")
    parser.add_argument('--data_dir', type=str, default='dataset', help='Path to CASIA 2.0 dataset root folder')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs (default 10 as in paper)')
    parser.add_argument('--batch_size', type=int, default=19, help='Batch size (default 19 as in paper)')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--resume', action='store_true', help='Resume training from last checkpoint')

    args = parser.parse_args()
    train_model(args.data_dir, args.epochs, args.batch_size, args.lr, resume=args.resume)
