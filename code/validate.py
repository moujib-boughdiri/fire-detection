import csv
from multiprocessing import freeze_support
from pathlib import Path
from ultralytics import YOLO
MODEL_PATH = r""
DATA_YAML  = r""
SPLIT      = "val"   
IMGSZ      = 640    
BATCH      = 16
DEVICE     = 0       
CONF       = 0.001     
IOU        = 0.6
PROJECT    = r""
NAME       = "fire_eval"

def evaluate(model, split):
    return model.val(
        data=DATA_YAML,
        split=split,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        conf=CONF,
        iou=IOU,
        plots=True,       
        save_json=False,
        save=True,        
        project=PROJECT,
        name=f"{NAME}_{split}",
        exist_ok=True,
        verbose=False,
    )


def report(results, split):
    box = results.box
    print(f"\n{'=' * 14} OVERALL METRICS ({split}) {'=' * 14}")
    print(f"Precision     : {box.mp:.4f}")
    print(f"Recall        : {box.mr:.4f}")
    print(f"mAP@0.5       : {box.map50:.4f}")
    print(f"mAP@0.75      : {box.map75:.4f}")
    print(f"mAP@0.5:0.95  : {box.map:.4f}")

    overall_f1 = 2 * box.mp * box.mr / (box.mp + box.mr + 1e-16)
    print(f"F1 (P/R mean) : {overall_f1:.4f}")

    speed = results.speed
    print(f"\nSpeed (ms/img): preprocess {speed['preprocess']:.1f} | "
          f"inference {speed['inference']:.1f} | postprocess {speed['postprocess']:.1f}")

    print(f"\n{'=' * 14} CLASS-WISE METRICS {'=' * 14}")
    header = f"{'Class':<12}{'Precision':>10}{'Recall':>10}{'F1':>10}{'mAP50':>10}{'mAP50-95':>10}"
    print(header)
    print("-" * len(header))

    rows = []

    for row, cls_id in enumerate(box.ap_class_index):
        name = results.names[int(cls_id)]
        p, r = box.p[row], box.r[row]
        f1 = box.f1[row]
        ap50, ap = box.ap50[row], box.ap[row]
        print(f"{name:<12}{p:>10.4f}{r:>10.4f}{f1:>10.4f}{ap50:>10.4f}{ap:>10.4f}")
        rows.append([name, p, r, f1, ap50, ap])

    return rows, [box.mp, box.mr, overall_f1, box.map50, box.map]


def save_csv(save_dir, rows, overall):
    csv_path = Path(save_dir) / "metrics_summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "precision", "recall", "f1", "mAP50", "mAP50-95"])
        for r in rows:
            w.writerow([r[0]] + [f"{v:.4f}" for v in r[1:]])
        w.writerow(["ALL"] + [f"{v:.4f}" for v in overall])
    print(f"\nSummary saved to: {csv_path}")


def main():
    model = YOLO(MODEL_PATH)
    results = evaluate(model, SPLIT)
    rows, overall = report(results, SPLIT)
    save_csv(results.save_dir, rows, overall)
    print(f"Plots and images saved in: {results.save_dir}")


if __name__ == "__main__":
    freeze_support()
    main()