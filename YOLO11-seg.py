from pathlib import Path

import torch
from ultralytics import YOLO

yaml_path = Path(__file__).parent / "foodseg103.yaml"
if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU not detected. Install a CUDA-enabled PyTorch build or check your drivers."
    )
training_device = 0

def main() -> None:
    model = YOLO("yolo11s-seg.pt")

    results = model.train(
        data=str(yaml_path),
        epochs=500,
        patience=50,
        imgsz=640,
        batch=32,
        device=training_device,
        workers=8,
        cos_lr=True,
        plots=True,
        project="runs/segment",
        name="yolo11s-foodseg103",
    )

    best_model = YOLO("runs/segment/yolo11s-foodseg103/weights/best.pt")
    best_model.export(format="onnx", imgsz=640, dynamic=True)


if __name__ == "__main__":
    main()