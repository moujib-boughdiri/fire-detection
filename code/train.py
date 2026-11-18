import os
from pathlib import Path
import yaml
import torch
from ultralytics import YOLO
os.environ['OMP_NUM_THREADS'] = '8'
torch.backends.cudnn.benchmark = True
DATASET_DIR = r""
DATA_YAML   = r""
WEIGHTS     = r""
PROJECT_DIR = r""
CLASS_NAMES = ['fire', 'smoke']
def ensure_data_yaml():
    """Create data.yaml if it does not exist."""
    if not os.path.isfile(DATA_YAML):
        data = {
            'path': DATASET_DIR.replace('\\', '/'),
            'train': 'train/images',
            'val': 'valid/images',
            'test': 'test/images',
            'nc': len(CLASS_NAMES),
            'names': CLASS_NAMES,
        }
        with open(DATA_YAML, 'w') as f:
            yaml.safe_dump(data, f, sort_keys=False)
        print(f"Created {DATA_YAML}")
    else:
        print(f"Using existing {DATA_YAML}")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Training requires GPU acceleration.")
    torch.cuda.empty_cache()

    ensure_data_yaml()
    if os.path.isfile(WEIGHTS):
        model = YOLO(WEIGHTS)
    else:
        print("Local weights not found, downloading yolov10m.pt ...")
        model = YOLO('yolov10m.pt')

    train_config = {
        'data': DATA_YAML,
        'epochs': 3,
        'imgsz': 640,
        'batch': 16,
        'name': 'speed_optimized',
        'project': PROJECT_DIR,
        'device': 0,
        'workers': 8,
        'cache': 'ram',
        'plots': True,
        'save_period': 5,
        'overlap_mask': False,
        'auto_augment': None,
        'optimizer': 'AdamW',
        'lr0': 0.001,
        'cos_lr': True,
        'warmup_epochs': 1,
        'close_mosaic': 0,
        'patience': 0,
        'box': 7.5,
        'cls': 0.5,
        'erasing': 0.0,
        'dfl': 1.5,
        'fliplr': 0.0,
        'mosaic': 0.0,
        'mixup': 0.0,
        'nbs': 64,
        'pretrained': True,
        'deterministic': False,
        'single_cls': False,
        'amp': True,
    }

    try:
        results = model.train(**train_config)

        if results:
            print("\nTraining completed successfully!")
            print(f"Best mAP50: {results.box.map50:.4f}")
            print(f"Best mAP50-95: {results.box.map:.4f}")

    except Exception as e:
        print(f"\nTraining failed: {str(e)}")
        torch.cuda.empty_cache()
        raise
if __name__ == "__main__":
    main()