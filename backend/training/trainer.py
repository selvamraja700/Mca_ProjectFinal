import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from models import ImprovedDenseNet
from .dataset import CasiaDataset


def train_model(data_dir, epochs=10, batch_size=19, lr=0.0001, save_path='models/densenet_forgery.pth', resume=False):
    """Train Improved DenseNet-121 on the CASIA v2.0 dataset."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

        except RuntimeError:
            import traceback
            traceback.print_exc()
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

        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'best_acc': best_acc
        }, checkpoint_path)

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), save_path)
