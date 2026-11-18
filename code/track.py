import os
from collections import Counter
from pathlib import Path

import torch
from ultralytics import YOLO

os.environ['OMP_NUM_THREADS'] = '8'

MODEL_PATH = r""
DATA_YAML  = r""
SOURCES    = [r"",r""]
PROJECT    = r""
TRACKER    = "bytetrack.yaml" 
CONF       = 0.35
IOU        = 0.5
IMGSZ      = 640
DEVICE     = 0
RUN_VAL    = True


VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}


def expand_source(src):
    p = Path(src)
    if p.is_dir():
        videos = sorted(f for f in p.iterdir() if f.suffix.lower() in VIDEO_EXT)
        if videos:
            return [str(v) for v in videos]
    return [src]


def track_one(model, src, name):
    print(f"\nTracking on: {src}")


    model.predictor = None

    results = model.track(
        source=src,
        tracker=TRACKER,
        persist=True,
        device=DEVICE,
        half=True,
        imgsz=IMGSZ,
        conf=CONF,
        iou=IOU,
        save=True,
        save_txt=False,
        show=False,
        line_width=1,
        project=PROJECT,
        name=name,
        exist_ok=True,
        stream=True,     
        verbose=False,
    )

    counts, ids = Counter(), {}
    n_frames, n_with_det = 0, 0
    for r in results:
        n_frames += 1
        if len(r.boxes):
            n_with_det += 1
            for c in r.boxes.cls.int().tolist():
                counts[r.names[c]] += 1
            if r.boxes.id is not None:
                for c, i in zip(r.boxes.cls.int().tolist(), r.boxes.id.int().tolist()):
                    ids.setdefault(r.names[c], set()).add(i)

    print(f"Processed {n_frames} frames, {n_with_det} with detections")
    for cls_name, n in counts.most_common():
        uniq = len(ids.get(cls_name, ()))
        print(f"  {cls_name}: {n} detections, {uniq} unique track IDs")
    print(f"Saved to: {PROJECT}\\{name}")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    model = YOLO(MODEL_PATH)

    k = 0
    for src in SOURCES:
        if not os.path.exists(src):
            print(f"Skipping missing source: {src}")
            continue
        for item in expand_source(src):
            track_one(model, item, f"run_{k}")
            k += 1

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