import os
import sys
import argparse
import time
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

try:
    import torch
    from torch.utils.data import DataLoader
    from torchvision import transforms
except ImportError:
    print("[!] PyTorch is not installed. Run: pip install torch torchvision pillow")
    sys.exit(1)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))
from model import ImprovedDenseNet
from train import CasiaDataset


def evaluate(data_dir='dataset', model_path='models/densenet_forgery.pth', batch_size=128, max_samples=None):
    # ── Device Setup ──────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 65)
    print("      IMAGE FORGERY DETECTION  –  FULL MODEL EVALUATION")
    print("=" * 65)

    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem  = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"[*] GPU  : {gpu_name}  ({gpu_mem:.1f} GB VRAM)")
        torch.backends.cudnn.benchmark = True   # fastest GPU kernels
    else:
        print("[*] Device: CPU  (no GPU detected)")

    # ── Model Weights ─────────────────────────────────────────────
    if not os.path.exists(model_path):
        print(f"\n[!] Model weights NOT found at '{model_path}'.")
        print("    Train the model first:  python train.py")
        return

    # ── Dataset ───────────────────────────────────────────────────
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    if not os.path.exists(data_dir):
        candidates = [
            os.path.join('data', 'raw', 'archive', 'CASIA2'),
            os.path.join('data', 'raw', 'archive', 'casia'),
            os.path.join('data', 'raw', 'CASIA2'),
            os.path.join('data', 'CASIA2'),
        ]
        found = False
        for p in candidates:
            if os.path.exists(p):
                print(f"[*] Auto-detected dataset at: '{p}'")
                data_dir = p
                found = True
                break
        if not found:
            print(f"[!] Dataset folder not found. Use: --data_dir <path>")
            return

    dataset = CasiaDataset(data_dir, transform=transform)
    if len(dataset) == 0:
        print(f"[!] No images found in '{data_dir}'.")
        return

    total_au = sum(1 for _, lbl in dataset.samples if lbl == 0)
    total_tp = sum(1 for _, lbl in dataset.samples if lbl == 1)

    # ── Optional balanced sub-sampling ───────────────────────────
    if max_samples and max_samples < len(dataset):
        import random
        au_idx = [i for i, (_, lbl) in enumerate(dataset.samples) if lbl == 0]
        tp_idx = [i for i, (_, lbl) in enumerate(dataset.samples) if lbl == 1]
        half   = max_samples // 2
        random.seed(42)
        selected = (random.sample(au_idx, min(half, len(au_idx))) +
                    random.sample(tp_idx, min(half, len(tp_idx))))
        random.shuffle(selected)
        dataset  = torch.utils.data.Subset(dataset, selected)
        sampled_au = min(half, len(au_idx))
        sampled_tp = min(half, len(tp_idx))
        print(f"[*] Balanced sample  : {sampled_au} Authentic + {sampled_tp} Tampered = {len(selected)} images")
    else:
        print(f"[*] Full dataset     : {total_au} Authentic + {total_tp} Tampered = {len(dataset)} images")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0, pin_memory=(device.type == 'cuda'))

    # ── Load Model ────────────────────────────────────────────────
    model = ImprovedDenseNet(pretrained=False).to(device)
    ckpt  = torch.load(model_path, map_location=device, weights_only=False)
    state = ckpt.get('model_state', ckpt)   # handle both checkpoint dicts and raw state dicts
    model.load_state_dict(state)
    model.eval()

    print(f"[*] Model weights    : {model_path}  ({os.path.getsize(model_path)/1e6:.1f} MB)")
    print(f"[*] Batch size       : {batch_size}  |  Total batches: {len(loader)}")
    print("=" * 65)
    print("  Starting evaluation – please wait …")
    print("-" * 65)

    # ── Inference Loop ────────────────────────────────────────────
    correct = 0
    au_correct = au_total = 0
    tp_correct = tp_total = 0
    total_samples = 0
    start = time.time()

    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits, _ = model(images)
            _, preds  = torch.max(logits, 1)

            correct       += (preds == labels).sum().item()
            total_samples += labels.size(0)

            au_mask = (labels == 0)
            tp_mask = (labels == 1)

            au_total   += au_mask.sum().item()
            tp_total   += tp_mask.sum().item()
            au_correct += (preds[au_mask] == 0).sum().item() if au_mask.any() else 0
            tp_correct += (preds[tp_mask] == 1).sum().item() if tp_mask.any() else 0

            # Progress every 50 batches
            if (i + 1) % 50 == 0 or (i + 1) == len(loader):
                run_acc  = correct / total_samples * 100
                elapsed  = time.time() - start
                eta      = elapsed / (i + 1) * (len(loader) - i - 1)
                print(f"  Batch {i+1:>4}/{len(loader)}  |"
                      f"  Images {total_samples:>6}/{len(dataset)}"
                      f"  |  Running Acc: {run_acc:6.2f}%"
                      f"  |  ETA: {eta:.0f}s",
                      flush=True)

    elapsed = time.time() - start

    # ── Metrics ───────────────────────────────────────────────────
    overall_acc = correct    / total_samples * 100 if total_samples > 0 else 0
    au_acc      = au_correct / au_total      * 100 if au_total      > 0 else 0
    tp_acc      = tp_correct / tp_total      * 100 if tp_total      > 0 else 0

    # Precision, Recall, F1 (TP = class 1 = Tampered)
    tp_val = tp_correct                          # True Positives  (Tampered predicted correctly)
    fp_val = au_total - au_correct               # False Positives (Authentic predicted as Tampered)
    fn_val = tp_total - tp_correct               # False Negatives (Tampered predicted as Authentic)

    precision = tp_val / (tp_val + fp_val) * 100 if (tp_val + fp_val) > 0 else 0
    recall    = tp_val / (tp_val + fn_val) * 100 if (tp_val + fn_val) > 0 else 0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

    print("\n" + "=" * 65)
    print("                    FINAL EVALUATION RESULTS")
    print("=" * 65)
    print(f"  Total Images Evaluated : {total_samples}  (done in {elapsed:.1f}s)")
    print(f"  Overall Accuracy       : {overall_acc:.2f}%  ({correct}/{total_samples})")
    print("-" * 65)
    print(f"  Authentic (Au) Acc     : {au_acc:.2f}%  ({au_correct}/{au_total})")
    print(f"  Tampered  (Tp) Acc     : {tp_acc:.2f}%  ({tp_correct}/{tp_total})")
    print("-" * 65)
    print(f"  Precision              : {precision:.2f}%")
    print(f"  Recall                 : {recall:.2f}%")
    print(f"  F1-Score               : {f1:.2f}%")
    print("=" * 65)

    if overall_acc >= 90:
        print("  ✅  Excellent model performance!")
    elif overall_acc >= 80:
        print("  🟡  Good model – consider more training epochs.")
    elif overall_acc >= 70:
        print("  🟠  Fair model – needs more training.")
    else:
        print("  🔴  Low accuracy – model may need retraining.")
    print("=" * 65)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Full GPU Accuracy Evaluation – Image Forgery Detection")
    parser.add_argument('--data_dir',    type=str,   default='dataset',
                        help='Path to dataset root (with Au/ and Tp/ folders)')
    parser.add_argument('--model_path',  type=str,   default='models/densenet_forgery.pth',
                        help='Path to saved model weights (.pth)')
    parser.add_argument('--batch_size',  type=int,   default=128,
                        help='Batch size for GPU inference (default 128; lower if GPU OOM)')
    parser.add_argument('--max_samples', type=int,   default=None,
                        help='Optional: limit to N balanced samples for a quick check')
    args = parser.parse_args()
    evaluate(args.data_dir, args.model_path, args.batch_size, args.max_samples)
