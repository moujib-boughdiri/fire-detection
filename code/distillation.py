import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer

ROOT = r""
DATA_DIR = r""
TEACHER_PATH = r""
DATA_YAML = r""

CLASSES = ["fire", "smoke"]
STUDENT_CFG = "yolov8n.yaml"
USE_PRETRAINED_STUDENT = True  

EPOCHS = 100
IMGSZ = 640
BATCH = 16


def write_data_yaml():
    lines = [
        f"train: {(DATA_DIR / 'train' / 'images').as_posix()}",
        f"val: {(DATA_DIR / 'val' / 'images').as_posix()}",
        f"nc: {len(CLASSES)}",
        f"names: {CLASSES}",
    ]
    DATA_YAML.write_text("\n".join(lines) + "\n")



def forward_feats(model, x):
    head = model.model[-1]
    y = []
    for m in model.model:
        if m.f != -1:
            x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
        x = m(x)
        y.append(x if m.i in model.save else None)
    feats = [y[j] for j in head.f]  # P3, P4, P5
    return x, feats



class DistillTrainer(DetectionTrainer):
    alpha = 1.0  

    def get_model(self, cfg=None, weights=None, verbose=True):
        student = super().get_model(cfg, weights, verbose)

        if USE_PRETRAINED_STUDENT and weights is None:
            student.load(YOLO("yolov8n.pt").model)  

        teacher = YOLO(str(TEACHER_PATH)).model.float().eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        self.teacher = teacher


        dummy = torch.zeros(1, 3, IMGSZ, IMGSZ)
        with torch.no_grad():
            student.eval()
            _, s_f = forward_feats(student, dummy)
            _, t_f = forward_feats(teacher, dummy)
            student.train()
        print("Student feature channels:", [f.shape[1] for f in s_f])
        print("Teacher feature channels:", [f.shape[1] for f in t_f])
        self.adapters = nn.ModuleList(
            [nn.Conv2d(s.shape[1], t.shape[1], 1) for s, t in zip(s_f, t_f)]
        )

        orig_loss = student.loss 

        def loss_with_distill(batch, preds=None):
            if preds is not None or not student.training:
                return orig_loss(batch, preds)

            img = batch["img"]
            dev = img.device
            if next(self.adapters.parameters()).device != dev:
                self.adapters.to(dev)
                self.teacher.to(dev)

            preds, s_feats = forward_feats(student, img)
            loss, items = orig_loss(batch, preds) 

            with torch.no_grad():
                _, t_feats = forward_feats(self.teacher, img)

            distill = sum(
                F.mse_loss(ad(sf).float(), tf.float())
                for ad, sf, tf in zip(self.adapters, s_feats, t_feats)
            )
            return loss + self.alpha * distill * img.shape[0], items

        student.loss = loss_with_distill
        return student

    def build_optimizer(self, *args, **kwargs):
        optimizer = super().build_optimizer(*args, **kwargs)
        optimizer.add_param_group({"params": list(self.adapters.parameters()), "weight_decay": 0.0})
        return optimizer

    def save_model(self):
        removed = []
        for m in (self.model, getattr(self.ema, "ema", None)):
            if m is not None and "loss" in m.__dict__:
                removed.append((m, m.__dict__.pop("loss")))
        try:
            super().save_model()
        finally:
            for m, fn in removed:
                m.__dict__["loss"] = fn

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["distill", "baseline"], default="distill")
    parser.add_argument("--alpha", type=float, default=1.0, help="distillation weight (try 0.1 - 2.0)")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch", type=int, default=BATCH)
    a = parser.parse_args()

    write_data_yaml()
    device = 0 if torch.cuda.is_available() else "cpu"

    common = dict(
        data=str(DATA_YAML),
        epochs=a.epochs,
        imgsz=IMGSZ,
        batch=a.batch,
        workers=2,
        device=device,
        project=str(ROOT / "runs"),
    )

    if a.mode == "baseline":
        model = YOLO("yolov8n.pt" if USE_PRETRAINED_STUDENT else STUDENT_CFG)
        model.train(name="student_baseline", **common)
        metrics = model.val()
    else:
        DistillTrainer.alpha = a.alpha
        trainer = DistillTrainer(overrides=dict(model=STUDENT_CFG, name=f"student_distill_a{a.alpha}", **common))
        trainer.train()
        metrics = YOLO(str(trainer.best)).val(data=str(DATA_YAML))

    print(f"[{a.mode}] mAP50: {metrics.box.map50:.4f} | mAP50-95: {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()