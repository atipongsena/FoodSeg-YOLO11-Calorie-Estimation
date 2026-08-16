import os
from pathlib import Path

import numpy as np
from PIL import Image
import cv2
from datasets import load_dataset
from tqdm.auto import tqdm

dataset_root = Path("foodseg103-yolo")
images_train_dir = dataset_root / "images" / "train"
images_val_dir   = dataset_root / "images" / "val"
labels_train_dir = dataset_root / "labels" / "train"
labels_val_dir   = dataset_root / "labels" / "val"

for d in [images_train_dir, images_val_dir, labels_train_dir, labels_val_dir]:
    d.mkdir(parents=True, exist_ok=True)

print("Dataset root:", dataset_root.resolve())


foodseg = load_dataset("EduardoPacheco/FoodSeg103")
print(foodseg)

# --------- ฟังก์ชัน: semantic mask -> YOLO11 segmentation labels ----------
def convert_split_to_yolo(split_name, img_dir: Path, label_dir: Path):
    """
    split_name: 'train' หรือ 'validation'
    img_dir, label_dir: path สำหรับเก็บรูปและ label (.txt)
    """
    ds = foodseg[split_name]

    for i, item in enumerate(tqdm(ds, desc=f"Converting {split_name}")):
        pil_img: Image.Image = item["image"]      # RGB image
        pil_mask: Image.Image = item["label"]     # single-channel mask
        mask = np.array(pil_mask)                 # H x W (ค่าคลาส 0..103)

        h, w = mask.shape

        # เซฟภาพเป็น jpg
        img_name = f"{split_name}_{i:06d}.jpg"
        img_path = img_dir / img_name
        pil_img.convert("RGB").save(img_path, format="JPEG", quality=95)

        # หา class ทั้งหมดในภาพ
        unique_classes = np.unique(mask)

        label_lines = []

        for cls_id in unique_classes:
            # 0 = background ใน FoodSeg103 → ไม่ต้องทำเป็น object
            if cls_id == 0:
                continue

            # แปลงเป็น binary mask ของคลาสนี้
            binary = (mask == cls_id).astype(np.uint8) * 255

            # หา contour สำหรับ instance แต่ละก้อน
            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                # ต้องมีอย่างน้อย 3 จุด
                if len(contour) < 3:
                    continue

                pts = contour.squeeze(1)  # shape (N, 2) ; (x, y) พิกเซล

                # normalize เป็น [0,1]
                xs = pts[:, 0] / float(w)
                ys = pts[:, 1] / float(h)

                # กันเคสหลุดกรอบ
                xs = np.clip(xs, 0, 1)
                ys = np.clip(ys, 0, 1)

                coords = []
                for x, y in zip(xs, ys):
                    coords.append(f"{x:.6f}")
                    coords.append(f"{y:.6f}")

                # YOLO class index: ลบ 1 เพราะ 0 คือ background
                yolo_cls = int(cls_id) - 1    # 0..102

                line = str(yolo_cls) + " " + " ".join(coords)
                label_lines.append(line)

        # เซฟเป็น .txt (หนึ่งไฟล์ต่อ 1 รูป)
        label_path = label_dir / (img_name.replace(".jpg", ".txt"))
        if label_lines:
            with open(label_path, "w") as f:
                f.write("\n".join(label_lines))
        else:
            # ถ้าไม่มี object (ปกติ FoodSeg103 มีหมด) จะสร้างไฟล์ว่าง
            label_path.touch()

    print(f"Done split: {split_name}, images={len(ds)}")

# รัน convert สำหรับ train / val
convert_split_to_yolo("train", images_train_dir, labels_train_dir)
convert_split_to_yolo("validation", images_val_dir, labels_val_dir)