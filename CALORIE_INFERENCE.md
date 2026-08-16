## Calorie-Aware YOLO11 Pipeline

### 1. Prepare the dataset

```powershell
python pre-dataset.py        # downloads FoodSeg103 and creates YOLO masks
```

This populates `foodseg103-yolo/images|labels/(train|val)` and updates `foodseg103.yaml` with absolute paths.

### 2. Train and export the segmentation model

```powershell
python YOLO11-seg.py         # trains yolo11n-seg.pt and exports ONNX
```

The script saves checkpoints to `runs/segment/yolo11n-foodseg103/` and exports `best.onnx` for mobile deployment.  
To target other devices, re-run the export step manually:

```powershell
from ultralytics import YOLO
YOLO("runs/segment/.../best.pt").export(format="onnx", dynamic=True)   # Android / desktop
YOLO("runs/segment/.../best.pt").export(format="coreml", imgsz=640)    # iOS
YOLO("runs/segment/.../best.pt").export(format="tflite", imgsz=640)    # Android / Edge TPU
```

### 3. Estimate calories on PC (Torch or ONNX backends)

```powershell
python calorie_inference.py ^
  --weights runs/segment/yolo11n-foodseg103/weights/best.pt ^
  --source samples/meal.jpg ^
  --metadata calorie_metadata.json ^
  --save-vis outputs/vis ^
  --save-json outputs/meal_calories.json ^
  --reference-object-cm 27 ^
  --reference-object-px 410
```

- `reference-object-*` lets you scale pixels to centimeters using a known plate diameter; alternatively pass `--scale-cm-per-px`.
- The script prints per-item calories, writes optional JSON summaries, and stores annotated images when `--save-vis` is provided.
- To run the same weights on mobile, load the exported ONNX/TFLite/CoreML file in your app and replicate the post-processing logic (mask area → grams → calories) from `calorie_inference.py`.

### 4. Customize nutrition assumptions

- `calorie_metadata.json` stores density (g/cm³) and calories per 100g per class. Update values to match your dietary database.
- If a class is missing, `calorie_inference.py` falls back to the default entry.
- `assumed_thickness_cm` controls the volume estimate (area × thickness). Adjust per cuisine or use multiple passes for layered dishes.

### 5. Quick sanity checks

- Run `python calorie_inference.py --source path/to/image --scale-cm-per-px 0.05` to verify the pipeline before integrating into mobile apps.
- Compare predicted totals with known nutrition labels to calibrate densities and calorie values.
