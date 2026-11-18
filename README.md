# Fire Detection

Fire and smoke detection using deep learning (YOLO) and computer vision. The project trains an object detection model on a labeled dataset and uses it to detect fire and smoke in images, videos, or a live camera feed.

## Features

- Detects `fire` and `smoke`
- Training on a clean, YOLO-format dataset
- Inference on images, videos, and webcam
- Reproducible environment through `requirements.txt`

## Project Structure

```
fire-detection/
├── code/              # Source code
├── dataset/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── valid/
│   │   ├── images/
│   │   └── labels/
│   └── test/
│       ├── images/
│       └── labels/
├── runs/              # Training and prediction outputs (not tracked)
├── weights/           # Trained model weights (not tracked)
├── requirements.txt
└── README.md
```

The `dataset`, `runs`, and `weights` folders are kept empty in the repository (using `.gitkeep` files) because their contents are too large for GitHub.

## Environment Setup

**Prerequisites:** Python 3.9 or newer and Git. A CUDA-capable NVIDIA GPU is recommended for training but not required for inference.

1. Clone the repository:

```
git clone https://github.com/moujib-boughdiri/fire-detection.git
cd fire-detection
```

2. Create and activate a virtual environment:

```
python -m venv venv

# Windows (PowerShell)
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

3. Upgrade pip and install the dependencies:

```
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

4. Verify the installation:

```
python -c "import ultralytics; ultralytics.checks()"
```

### Tips

- Always use `python -m pip` instead of `pip`. It guarantees the packages are installed in the active environment.
- Never move or rename the `venv` folder after creating it, because it stores absolute paths. If you move the project, delete `venv` and recreate it.
- To use your GPU, install the CUDA build of PyTorch from https://pytorch.org before installing the other requirements.
- If PowerShell blocks activation, run this once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

## Dataset

The dataset is cleaned and ready to use, and is hosted on Kaggle:

https://www.kaggle.com/datasets/moujibboughdiri/fire-data-2

1. Download and extract it.
2. Place the `train`, `valid`, and `test` folders inside `dataset/`, following the structure above.
3. Create a `data.yaml` file in the project root (adjust class names to match your labels):

```
path: dataset
train: train/images
val: valid/images
test: test/images

names:
  0: fire
  1: smoke
```

Labels are in YOLO format (one `.txt` file per image).

## Training

Train a model with the Ultralytics CLI:

```
yolo detect train data=data.yaml model=yolov8n.pt epochs=100 imgsz=640 batch=16 project=runs name=fire_train
```

Tips for a high-quality model:

- Start from a pretrained model (`yolov8n.pt` for speed, `yolov8s.pt` or `yolov8m.pt` for accuracy).
- Train for 100 to 300 epochs and let early stopping (`patience=50`) decide when to stop.
- Keep `imgsz=640` or higher, since small flames and thin smoke benefit from resolution.
- Make sure the train, valid, and test splits do not share the same images or near-duplicates.
- Check `runs/fire_train/results.png` and the confusion matrix, then add more data for the classes with weak recall.
- Evaluate on the test set before deploying (see below).

The best model is saved at `runs/fire_train/weights/best.pt`. Copy it to the `weights/` folder.

## Evaluation

```
yolo detect val model=weights/best.pt data=data.yaml split=test
```

## Running Detection

On an image:

```
yolo detect predict model=weights/best.pt source=path/to/image.jpg conf=0.4 project=runs name=predict
```

On a video:

```
yolo detect predict model=weights/best.pt source=path/to/video.mp4 conf=0.4 save=True
```

On a webcam:

```
yolo detect predict model=weights/best.pt source=0 conf=0.4 show=True
```

Results are saved in the `runs/` folder.

## Pretrained Model

The trained model weights are not included in this repository and are not publicly available. Following the steps in this README (clean dataset, pretrained base model, sufficient epochs, and validation on the test set), you can train your own model with the same approach.

## Tech Stack

- Python
- PyTorch
- Ultralytics YOLO
- OpenCV

## Author

Created by [moujib-boughdiri](https://github.com/moujib-boughdiri).
