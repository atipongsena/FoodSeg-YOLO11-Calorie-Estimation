from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
import yaml
from ultralytics import YOLO


@dataclass
class FoodMetadata:
    calories_per_100g: float
    density_g_per_cm3: float


@dataclass
class ItemCalorieEstimate:
    class_id: int
    class_name: str
    pixel_area: float
    area_cm2: float
    weight_grams: float
    calories: float

    def to_dict(self) -> Dict[str, float]:
        payload = asdict(self)
        payload["calories"] = round(self.calories, 2)
        payload["weight_grams"] = round(self.weight_grams, 2)
        payload["area_cm2"] = round(self.area_cm2, 2)
        payload["pixel_area"] = round(self.pixel_area, 2)
        return payload


@dataclass
class ImageCalorieSummary:
    source: str
    estimates: List[ItemCalorieEstimate]

    @property
    def total_calories(self) -> float:
        return sum(item.calories for item in self.estimates)

    @property
    def total_weight(self) -> float:
        return sum(item.weight_grams for item in self.estimates)

    def to_dict(self) -> Dict:
        return {
            "source": self.source,
            "total_calories": round(self.total_calories, 2),
            "total_weight_grams": round(self.total_weight, 2),
            "items": [item.to_dict() for item in self.estimates],
        }


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


class CalorieEstimator:
    def __init__(
        self,
        weights_path: Path,
        data_yaml: Path,
        metadata_path: Path,
        scale_estimator: ScaleEstimator,
        thickness_cm: Optional[float],
        device: Optional[str],
        imgsz: int,
        conf: float,
    ) -> None:
        self.model = YOLO(str(weights_path))
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.scale_estimator = scale_estimator
        self.calorie_metadata = CalorieMetadata(metadata_path)
        self.thickness_cm = (
            thickness_cm if thickness_cm and thickness_cm > 0 else self.calorie_metadata.assumed_thickness_cm
        )
        self.class_names = self._load_class_names(data_yaml)

    @staticmethod
    def _load_class_names(data_yaml: Path) -> List[str]:
        data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
        names = data.get("names", {})
        if isinstance(names, dict):
            return [names[k] for k in sorted(names.keys())]
        return list(names)

    def estimate(self, source: str) -> List[ImageCalorieSummary]:
        predictions = self.model.predict(
            source,
            imgsz=self.imgsz,
            conf=self.conf,
            device=self.device,
            stream=True,
            verbose=False,
        )
        summaries: List[ImageCalorieSummary] = []
        for result in predictions:
            summary = self._summarize_result(result)
            summaries.append(summary)
        return summaries

    def _summarize_result(self, result) -> ImageCalorieSummary:
        if result.masks is None or result.boxes is None or len(result.boxes) == 0:
            return ImageCalorieSummary(source=str(result.path), estimates=[])

        masks = result.masks.data.cpu().numpy()  # shape: (instances, H, W)
        cls_ids = result.boxes.cls.int().tolist()
        scale_cm_per_px = self.scale_estimator.resolve(result.orig_shape)
        area_scale = scale_cm_per_px ** 2

        estimates: List[ItemCalorieEstimate] = []
        for idx, (cls_id, mask_array) in enumerate(zip(cls_ids, masks)):
            class_name = self.class_names[cls_id]
            metadata = self.calorie_metadata.get(class_name)
            pixel_area = float(mask_array.sum())
            area_cm2 = pixel_area * area_scale
            volume_cm3 = area_cm2 * self.thickness_cm
            weight_grams = volume_cm3 * metadata.density_g_per_cm3
            calories = (weight_grams * metadata.calories_per_100g) / 100.0
            estimates.append(
                ItemCalorieEstimate(
                    class_id=cls_id,
                    class_name=class_name,
                    pixel_area=pixel_area,
                    area_cm2=area_cm2,
                    weight_grams=weight_grams,
                    calories=calories,
                )
            )
        return ImageCalorieSummary(source=str(result.path), estimates=estimates)

    def save_visualizations(
        self,
        summaries: List[ImageCalorieSummary],
        output_dir: Optional[Path],
    ) -> None:
        if not output_dir:
            return
        output_dir.mkdir(parents=True, exist_ok=True)
        predictions = self.model.predict(
            [summary.source for summary in summaries],
            imgsz=self.imgsz,
            conf=self.conf,
            device=self.device,
            stream=False,
            verbose=False,
        )
        for result, summary in zip(predictions, summaries):
            annotated = result.plot()  # BGR image
            if result.boxes is not None and summary.estimates:
                for estimate, box in zip(summary.estimates, result.boxes.xyxy.tolist()):
                    x1, y1 = int(box[0]), int(box[1])
                    text = f"{estimate.class_name}: {estimate.calories:.0f} kcal"
                    cv2.putText(
                        annotated,
                        text,
                        (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
            out_path = output_dir / f"{Path(result.path).stem}_calorie.jpg"
            cv2.imwrite(str(out_path), annotated)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate calories from YOLO11 segmentation outputs."
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("runs/segment/yolo11s-foodseg103/weights/best.pt"),
        help="Path to trained YOLO weights (.pt or .onnx).",
    )
    parser.add_argument(
        "--data-yaml",
        type=Path,
        default=Path("foodseg103.yaml"),
        help="Dataset YAML containing class names.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("calorie_metadata.json"),
        help="JSON file with calorie/density assumptions.",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Image, directory, or glob to run inference on.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Inference device (e.g., 'cuda:0', 'cpu', '0').",
    )
    parser.add_argument(
        "--imgsz", type=int, default=640, help="Inference image size."
    )
    parser.add_argument(
        "--conf", type=float, default=0.4, help="Confidence threshold."
    )
    parser.add_argument(
        "--scale-cm-per-px",
        type=float,
        default=None,
        help="Explicit centimeters per pixel scale. Overrides reference options.",
    )
    parser.add_argument(
        "--reference-object-cm",
        type=float,
        default=None,
        help="Known size (cm) of an object in the scene (e.g., plate diameter).",
    )
    parser.add_argument(
        "--reference-object-px",
        type=float,
        default=None,
        help="Pixel width of the reference object in the image.",
    )
    parser.add_argument(
        "--thickness-cm",
        type=float,
        default=None,
        help="Assumed average food thickness in centimeters.",
    )
    parser.add_argument(
        "--save-vis",
        type=Path,
        default=None,
        help="Optional directory to save annotated calorie visualizations.",
    )
    parser.add_argument(
        "--save-json",
        type=Path,
        default=None,
        help="Optional path to save calorie summaries as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scale_estimator = ScaleEstimator(
        explicit_scale=args.scale_cm_per_px,
        reference_object_cm=args.reference_object_cm,
        reference_object_px=args.reference_object_px,
    )
    estimator = CalorieEstimator(
        weights_path=args.weights,
        data_yaml=args.data_yaml,
        metadata_path=args.metadata,
        scale_estimator=scale_estimator,
        thickness_cm=args.thickness_cm,
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
    )

    summaries = estimator.estimate(args.source)
    for summary in summaries:
        print(f"\nSource: {summary.source}")
        print(f"  Total calories: {summary.total_calories:.1f} kcal")
        print(f"  Total weight : {summary.total_weight:.1f} g")
        for item in summary.estimates:
            print(
                f"    - {item.class_name:<20} {item.calories:6.1f} kcal | {item.weight_grams:6.1f} g | area {item.area_cm2:7.1f} cm^2"
            )

    if args.save_json:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        json_payload = [summary.to_dict() for summary in summaries]
        args.save_json.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")
        print(f"\nSaved calorie summary to {args.save_json}")

    if args.save_vis:
        estimator.save_visualizations(summaries, args.save_vis)
        print(f"Saved annotated visualizations to {args.save_vis}")


if __name__ == "__main__":
    main()

