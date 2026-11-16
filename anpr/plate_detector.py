"""Plate localisation: trained YOLO plate model, with an EasyOCR fallback."""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np

from .ocr import PlateOCR, PlateRead


class PlateFinder:
    """Finds and reads the plate inside each vehicle crop.

    * If a trained plate detector (`weights/plate_yolov8n.pt`) exists, YOLO
      localises the plate and EasyOCR reads only that region (fast, accurate).
    * Otherwise EasyOCR's own text detector scans the vehicle crop and the
      most plate-like text region is kept (works out of the box, slower).
    """

    def __init__(self, ocr: PlateOCR, plate_model_path: Optional[str], conf: float = 0.25,
                 device: str = "", min_plate_width: int = 30, imgsz: int = 416):
        self.ocr = ocr
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        self.min_plate_width = min_plate_width
        self.model = None
        if plate_model_path and os.path.isfile(plate_model_path):
            from ultralytics import YOLO

            self.model = YOLO(plate_model_path)
            print(f"[plate] using YOLO plate detector: {plate_model_path}")
        else:
            print("[plate] no plate model found -> using EasyOCR text-detection fallback "
                  "(train one with train_plate_detector.py for better accuracy)")

    @property
    def mode(self) -> str:
        return "yolo" if self.model is not None else "ocr-fallback"

    def read(self, vehicle_crops: List[np.ndarray]) -> List[Optional[PlateRead]]:
        if not vehicle_crops:
            return []
        if self.model is None:
            return [self.ocr.find_and_read(c) for c in vehicle_crops]

        kwargs = dict(conf=self.conf, imgsz=self.imgsz, verbose=False)
        if self.device:
            kwargs["device"] = self.device
        results = self.model.predict(vehicle_crops, **kwargs)
        out: List[Optional[PlateRead]] = []
        for crop, res in zip(vehicle_crops, results):
            best: Optional[PlateRead] = None
            if res.boxes is not None and len(res.boxes):
                # try the two most confident plate boxes
                order = np.argsort(-res.boxes.conf.cpu().numpy())[:2]
                xyxy = res.boxes.xyxy.cpu().numpy()
                for i in order:
                    x1, y1, x2, y2 = xyxy[i].astype(int)
                    if x2 - x1 < self.min_plate_width:
                        continue
                    # small padding helps OCR with characters touching the border
                    pw, ph = int(0.06 * (x2 - x1)), int(0.12 * (y2 - y1))
                    h, w = crop.shape[:2]
                    px1, py1 = max(x1 - pw, 0), max(y1 - ph, 0)
                    px2, py2 = min(x2 + pw, w), min(y2 + ph, h)
                    plate = crop[py1:py2, px1:px2].copy()
                    read = self.ocr.read_plate(plate)
                    if read is None:
                        continue
                    read.box = (int(x1), int(y1), int(x2), int(y2))
                    if best is None or read.score > best.score:
                        best = read
            out.append(best)
        return out
