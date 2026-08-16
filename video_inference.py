from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

DEFAULT_SOURCE = "Steak is just better at home.mp4"


@dataclass
class FoodMetadata:
    calories_per_100g: float
    density_g_per_cm3: float


class CalorieMetadata:
    def __init__(self, metadata_path: Path):
        raw = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.assumed_thickness_cm: float = float(raw.get("assumed_thickness_cm", 1.5))
        self.default = self._parse_entry(raw.get("default", {}))
        classes = raw.get("classes", {})
        self.class_lookup = {
            self._normalize_name(name): self._parse_entry(values)
            for name, values in classes.items()
        }

    @staticmethod
    def _normalize_name(name: str) -> str:
        return name.strip().lower()

    @staticmethod
    def _parse_entry(values: Dict) -> FoodMetadata:
        return FoodMetadata(
            calories_per_100g=float(values.get("calories_per_100g", 180.0)),
            density_g_per_cm3=float(values.get("density_g_per_cm3", 0.75)),
        )

    def get(self, class_name: str) -> FoodMetadata:
        return self.class_lookup.get(self._normalize_name(class_name), self.default)


class ScaleEstimator:
    def __init__(
        self,
        explicit_scale: Optional[float],
        reference_object_cm: Optional[float],
        reference_object_px: Optional[float],
        fallback_object_cm: float = 24.0,
    ) -> None:
        self.explicit_scale = explicit_scale
        self.reference_object_cm = reference_object_cm
        self.reference_object_px = reference_object_px
        self.fallback_object_cm = fallback_object_cm

    def resolve(self, image_shape: Sequence[int]) -> float:
        if self.explicit_scale and self.explicit_scale > 0:
            return self.explicit_scale
        if (
            self.reference_object_cm
            and self.reference_object_px
            and self.reference_object_px > 0
        ):
            return self.reference_object_cm / self.reference_object_px
        shorter_side = float(min(image_shape[0], image_shape[1]))
        return self.fallback_object_cm / shorter_side


def load_class_names(data_yaml: Path) -> List[str]:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = data.get("names", {})
    if isinstance(names, dict):
        return [names[k] for k in sorted(names.keys())]
    return list(names)


def estimate_calories(
    result,
    class_names: List[str],
    calorie_metadata: CalorieMetadata,
    scale_estimator: ScaleEstimator,
    thickness_cm: Optional[float],
) -> Dict[str, float]:
    if result.masks is None or result.boxes is None or len(result.boxes) == 0:
        return {"total_calories": 0.0, "total_weight": 0.0, "items": []}

    masks = result.masks.data.cpu().numpy()
    cls_ids = result.boxes.cls.int().tolist()
    boxes = result.boxes.xyxy.tolist()
    scale_cm_per_px = scale_estimator.resolve(result.orig_shape)
    area_scale = scale_cm_per_px ** 2
    thickness = (
        thickness_cm if thickness_cm and thickness_cm > 0 else calorie_metadata.assumed_thickness_cm
    )

    total_calories = 0.0
    total_weight = 0.0
    items = []

    for cls_id, mask_array, box in zip(cls_ids, masks, boxes):
        class_name = class_names[cls_id]
        metadata = calorie_metadata.get(class_name)
        pixel_area = float(mask_array.sum())
        area_cm2 = pixel_area * area_scale
        volume_cm3 = area_cm2 * thickness
        weight_grams = volume_cm3 * metadata.density_g_per_cm3
        calories = (weight_grams * metadata.calories_per_100g) / 100.0
        total_calories += calories
        total_weight += weight_grams
        items.append(
            {
                "class": class_name,
                "calories": calories,
                "weight": weight_grams,
                "area_cm2": area_cm2,
                "box": box,
            }
        )

    return {
        "total_calories": total_calories,
        "total_weight": total_weight,
        "items": items,
    }


def overlay_item_calories(frame: np.ndarray, summary: Dict[str, float]) -> None:
    for item in summary["items"]:
        x1, y1, x2, y2 = map(int, item["box"])
        label = f"{item['calories']:.0f} kcal"
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        text_width, text_height = text_size
        text_x = x1
        text_y = min(frame.shape[0] - 5, y2 + text_height + 10)

        cv2.rectangle(
            frame,
            (text_x, text_y - text_height - 6),
            (text_x + text_width + 6, text_y + 4),
            (0, 0, 0),
            thickness=-1,
        )
        cv2.putText(
            frame,
            label,
            (text_x + 3, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )


def overlay_total_text(
    frame: np.ndarray,
    summary: Dict[str, float],
    position: tuple[int, int] = (20, 40),
) -> None:
    text = f"Total: {summary['total_calories']:.1f} kcal | {summary['total_weight']:.1f} g"
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO11 segmentation on video with optional calorie overlay."
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("runs/segment/yolo11s-foodseg103/weights/best.pt"),
        help="Path to trained YOLO weights.",
    )
    parser.add_argument(
        "--data-yaml",
        type=Path,
        default=Path("foodseg103.yaml"),
        help="Dataset YAML describing class names.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Video file path, directory, or webcam index. Falls back to DEFAULT_SOURCE.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/video_result.mp4"),
        help="Where to save annotated video.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold.")
    parser.add_argument("--device", type=str, default=None, help="Device string.")
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("calorie_metadata.json"),
        help="Calorie metadata JSON (defaults to calorie_metadata.json in project root).",
    )
    parser.add_argument(
        "--scale-cm-per-px",
        type=float,
        default=None,
        help="Explicit centimeters-per-pixel scale.",
    )
    parser.add_argument(
        "--reference-object-cm",
        type=float,
        default=None,
        help="Real size (cm) of a known object in the scene.",
    )
    parser.add_argument(
        "--reference-object-px",
        type=float,
        default=None,
        help="Pixel width of that known object.",
    )
    parser.add_argument(
        "--thickness-cm",
        type=float,
        default=None,
        help="Assumed food thickness. Overrides metadata default if provided.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display video in a window while processing.",
    )
    parser.add_argument(
        "--vid-stride",
        type=int,
        default=1,
        help="Frame stride for inference (use >1 to skip frames).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    source = args.source or DEFAULT_SOURCE
    model = YOLO(str(args.weights))
    class_names = load_class_names(args.data_yaml)

    calorie_metadata = (
        CalorieMetadata(args.metadata) if args.metadata and args.metadata.exists() else None
    )
    scale_estimator = ScaleEstimator(
        explicit_scale=args.scale_cm_per_px,
        reference_object_cm=args.reference_object_cm,
        reference_object_px=args.reference_object_px,
    )

    output_parent = args.output.parent
    output_parent.mkdir(parents=True, exist_ok=True)

    predictions = model.predict(
        source,
        stream=True,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        vid_stride=args.vid_stride,
        verbose=True,
    )

    writer: Optional[cv2.VideoWriter] = None

    try:
        for result in predictions:
            frame = result.plot()  # ndarray (BGR)

            if calorie_metadata:
                summary = estimate_calories(
                    result,
                    class_names=class_names,
                    calorie_metadata=calorie_metadata,
                    scale_estimator=scale_estimator,
                    thickness_cm=args.thickness_cm,
                )
                overlay_item_calories(frame, summary)
                overlay_total_text(frame, summary)

            if writer is None:
                fps = result.fps if hasattr(result, "fps") and result.fps else 25
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(
                    str(args.output),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height),
                )

            writer.write(frame)

            if args.show:
                cv2.imshow("YOLO11 Video", frame)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
                    break
    finally:
        if writer:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()

    print(f"Processed source: {source}")
    print(f"Saved annotated video to {args.output}")


if __name__ == "__main__":
    main()

