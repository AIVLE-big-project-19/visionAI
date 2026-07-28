from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

from .config import Settings


# Kept from the existing notebook: only these detected classes are candidates.
INSTALLABLE = {"building", "parking_lot", "land"}


@dataclass(frozen=True)
class Extent3857:
    min_x: float
    min_y: float
    max_x: float
    max_y: float

    def validate(self) -> None:
        if self.min_x >= self.max_x or self.min_y >= self.max_y:
            raise ValueError("extent3857 must satisfy minX < maxX and minY < maxY.")


class YoloSegmentationService:
    """Owns the single in-memory model and performs no request persistence."""

    def __init__(self, settings: Settings) -> None:
        if not settings.model_path.is_file():
            raise RuntimeError(f"YOLO model file does not exist: {settings.model_path}")

        self._settings = settings
        self._model = YOLO(str(settings.model_path))
        # A single loaded model is shared. Serializing predict avoids concurrent
        # access to the same PyTorch model instance in multi-request deployments.
        self._predict_lock = Lock()

    def predict(self, image: np.ndarray, extent: Extent3857) -> list[dict[str, Any]]:
        height, width = image.shape[:2]
        if height == 0 or width == 0:
            raise ValueError("The uploaded image has no pixels.")

        with self._predict_lock:
            results = self._model.predict(
                image,
                conf=self._settings.min_confidence,
                verbose=False,
            )

        result = results[0]
        if result.masks is None or result.boxes is None:
            return []

        polygons = result.masks.xy
        classes = result.boxes.cls.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        candidates: list[dict[str, Any]] = []

        for polygon, class_id, confidence in zip(polygons, classes, confidences):
            candidate_type = result.names[int(class_id)]
            if candidate_type not in INSTALLABLE or float(confidence) < self._settings.min_confidence:
                continue

            # This is the same area filter used in the notebook.
            polygon_int = polygon.astype(np.int32)
            pixel_area = float(cv2.contourArea(polygon_int))
            if pixel_area < self._settings.min_pixel_area:
                continue

            candidates.append(
                {
                    "candidate_type": candidate_type,
                    "confidence": round(float(confidence), 4),
                    # Coordinates are converted from image pixels to EPSG:3857.
                    "polygon": self._to_3857_polygon(polygon, width, height, extent),
                    "pixel_area": round(pixel_area, 2),
                    "real_area": round(self._real_area(pixel_area, width, height, extent), 2),
                    "model_version": self._settings.model_version,
                }
            )

        return candidates

    @staticmethod
    def _real_area(pixel_area: float, width: int, height: int, extent: Extent3857) -> float:
        meters_per_pixel_x = (extent.max_x - extent.min_x) / width
        meters_per_pixel_y = (extent.max_y - extent.min_y) / height
        return pixel_area * meters_per_pixel_x * meters_per_pixel_y

    @staticmethod
    def _to_3857_polygon(
        polygon: np.ndarray, width: int, height: int, extent: Extent3857
    ) -> list[list[float]]:
        converted: list[list[float]] = []
        for pixel_x, pixel_y in polygon:
            map_x = extent.min_x + (float(pixel_x) / width) * (extent.max_x - extent.min_x)
            # Image Y increases downward; Web Mercator Y increases upward.
            map_y = extent.max_y - (float(pixel_y) / height) * (extent.max_y - extent.min_y)
            converted.append([round(map_x, 3), round(map_y, 3)])
        return converted
