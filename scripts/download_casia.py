import os
import zipfile
import urllib.request

DATASET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset'))

def setup_dataset_structure():
    au_dir = os.path.join(DATASET_DIR, 'au')
    tp_dir = os.path.join(DATASET_DIR, 'tp')
    
    os.makedirs(au_dir, exist_ok=True)
    os.makedirs(tp_dir, exist_ok=True)
    
    print(f"[+] Created CASIA v2.0 directory structure at:")
    print(f"    - Authentic (Genuine) Images: {au_dir}")
    print(f"    - Tampered (Forged) Images:   {tp_dir}")
    print("\nTo train on the full CASIA v2.0 database:")
    print("1. Download CASIA v2.0 dataset from Kaggle or Official Repo:")
    print("   https://www.kaggle.com/datasets/sophatvathana/casia-dataset")
    print("2. Extract Authentic images into 'dataset/au/' and Tampered images into 'dataset/tp/'")
    print("3. Execute training script: python train.py --data_dir dataset --epochs 10")

if __name__ == '__main__':
    setup_dataset_structure()
