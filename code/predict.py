import os
from collections import Counter

import torch
from ultralytics import YOLO

os.environ['OMP_NUM_THREADS'] = '8'

MODEL_PATH = r""
DATA_YAML  = r""
SOURCES    = [r"",r""]
PROJECT    = r""
CONF       = 0.35
IOU        = 0.5       
IMGSZ      = 640
BATCH      = 16        
DEVICE     = 0
RUN_VAL    = True      


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    model = YOLO(MODEL_PATH)

    for i, src in enumerate(SOURCES):
        if not os.path.exists(src):
            print(f"Skipping missing source: {src}")
            continue

        print(f"\nRunning prediction on: {src}")
        results = model.predict(
            source=src,
            device=DEVICE,
            batch=BATCH,
            imgsz=IMGSZ,
            half=True,
            save=True,
            save_txt=False,
            show=False,
            conf=CONF,
            iou=IOU,
            agnostic_nms=False,
            line_width=1,
            project=PROJECT,
            name=f"run_{i}",
            exist_ok=True,
            stream=True,    
            verbose=False,
        )

        counts, n_images, n_with_det = Counter(), 0, 0
        for r in results:
            n_images += 1
            if len(r.boxes):
                n_with_det += 1
                for c in r.boxes.cls.int().tolist():
                    counts[r.names[c]] += 1

        print(f"Processed {n_images} images, {n_with_det} with detections")
        for name, n in counts.most_common():
            print(f"  {name}: {n} detections")
        print(f"Saved to: {PROJECT}\\run_{i}")

    if RUN_VAL:
        print("\nEvaluating on the test split...")
        res = model.val(
            data=DATA_YAML,
            split="test",
            imgsz=IMGSZ,
            device=DEVICE,
            plots=True,
            project=PROJECT,
            name="test_metrics",
            exist_ok=True,
        )
        print(f"Precision: {res.box.mp:.4f} | Recall: {res.box.mr:.4f}")
        print(f"mAP50: {res.box.map50:.4f} | mAP50-95: {res.box.map:.4f}")


if __name__ == "__main__":
    main()