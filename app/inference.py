from __future__ import annotations

import base64
from dataclasses import dataclass
from threading import Lock
from typing import Any

import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
from ultralytics import YOLO

from .config import Settings


# Kept from the existing notebook: only these detected classes are candidates.
INSTALLABLE = {"building", "parking_lot", "land"}

# BGR colors per candidate_type, used for the annotated overlay image.
OVERLAY_COLORS: dict[str, tuple[int, int, int]] = {
    "building": (255, 0, 0),
    "land": (0, 200, 0),
    "parking_lot": (0, 200, 255),
}
OVERLAY_FILL_ALPHA = 0.35

# Solar panel dimensions in metres. The calculated area is used for the
# estimated panel count, so changing either dimension updates the simulation.
PANEL_WIDTH_M = 2.465
PANEL_HEIGHT_M = 1.134
PANEL_AREA_M2 = PANEL_WIDTH_M * PANEL_HEIGHT_M  # 2.7953
PACKING_FACTOR = 0.90


WATERMARK_WIDTH_FRACTION = 0.24
WATERMARK_HEIGHT_FRACTION = 0.10
WATERMARK_OVERLAP_THRESHOLD = 0.5


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

    def predict(
        self, image: np.ndarray, extent: Extent3857
    ) -> tuple[list[dict[str, Any]], str]:
        """Returns (candidates, annotated_image_base64_png)."""
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
            return [], self._encode_annotated(image, [])

        polygons = result.masks.xy
        classes = result.boxes.cls.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        candidates: list[dict[str, Any]] = []

        # The updated model logic uses every detected road/building polygon as
        # the distance target. These are not limited to installable candidates.
        road_polygons_px: list[BaseGeometry] = []
        building_polygons_px: list[BaseGeometry] = []
        road_polygons_m: list[BaseGeometry] = []
        building_polygons_m: list[BaseGeometry] = []

        for polygon, class_id in zip(polygons, classes):
            detected_type = result.names[int(class_id)]
            if self._overlaps_watermark(polygon.astype(np.int32), width, height):
                continue
            pixel_geometry = self._to_geometry(polygon)
            map_geometry = self._to_geometry(
                self._to_3857_polygon(polygon, width, height, extent)
            )
            if pixel_geometry is None or map_geometry is None:
                continue

            if detected_type == "road":
                road_polygons_px.append(pixel_geometry)
                road_polygons_m.append(map_geometry)
            elif detected_type == "building":
                building_polygons_px.append(pixel_geometry)
                building_polygons_m.append(map_geometry)

        annotated_polygons_px: list[tuple[np.ndarray, str]] = []

        for polygon, class_id, confidence in zip(polygons, classes, confidences):
            candidate_type = result.names[int(class_id)]
            if candidate_type not in INSTALLABLE or float(confidence) < self._settings.min_confidence:
                continue

            # This is the same area filter used in the notebook.
            polygon_int = polygon.astype(np.int32)
            if self._overlaps_watermark(polygon_int, width, height):
                continue
            pixel_area = float(cv2.contourArea(polygon_int))
            if pixel_area < self._settings.min_pixel_area:
                continue

            annotated_polygons_px.append((polygon_int, candidate_type))

            shape_info = self._calculate_shape_features(polygon)
            real_area = self._real_area(pixel_area, width, height, extent)
            usable_area = (
                real_area * shape_info["shape_efficiency"] * PACKING_FACTOR
            )
            estimated_panel_count = int(usable_area / PANEL_AREA_M2)

            distance_to_road_px: float | None = None
            distance_to_building_px: float | None = None
            distance_to_road_m: float | None = None
            distance_to_building_m: float | None = None

            # Retain the supplied logic: distances are relevant only to land.
            if candidate_type == "land":
                pixel_geometry = self._to_geometry(polygon)
                map_geometry = self._to_geometry(
                    self._to_3857_polygon(polygon, width, height, extent)
                )
                if pixel_geometry is not None and map_geometry is not None:
                    if road_polygons_px:
                        distance_to_road_px = min(
                            pixel_geometry.distance(road) for road in road_polygons_px
                        )
                        distance_to_road_m = min(
                            map_geometry.distance(road) for road in road_polygons_m
                        )
                    if building_polygons_px:
                        distance_to_building_px = min(
                            pixel_geometry.distance(building)
                            for building in building_polygons_px
                        )
                        distance_to_building_m = min(
                            map_geometry.distance(building)
                            for building in building_polygons_m
                        )

            candidates.append(
                {
                    "candidate_type": candidate_type,
                    "confidence": round(float(confidence), 4),
                    # Coordinates are converted from image pixels to EPSG:3857.
                    "polygon": self._to_3857_polygon(polygon, width, height, extent),
                    "pixel_area": round(pixel_area, 2),
                    "real_area": round(real_area, 2),
                    "distance_to_road_px": self._round_or_none(distance_to_road_px),
                    "distance_to_building_px": self._round_or_none(distance_to_building_px),
                    "distance_to_road_m": self._round_or_none(distance_to_road_m),
                    "distance_to_building_m": self._round_or_none(distance_to_building_m),
                    "shape_score": shape_info["shape_score"],
                    "shape_grade": shape_info["shape_grade"],
                    "shape_efficiency": shape_info["shape_efficiency"],
                    "recommended_layout": shape_info["recommended_layout"],
                    "usable_area": round(usable_area, 2),
                    "estimated_panel_count": estimated_panel_count,
                    "model_version": self._settings.model_version,
                }
            )

        return candidates, self._encode_annotated(image, annotated_polygons_px)

    @classmethod
    def _encode_annotated(
        cls, image: np.ndarray, polygons_px: list[tuple[np.ndarray, str]]
    ) -> str:
        annotated = image.copy()
        if polygons_px:
            overlay = image.copy()
            for polygon_int, candidate_type in polygons_px:
                color = OVERLAY_COLORS.get(candidate_type, (255, 255, 255))
                cv2.fillPoly(overlay, [polygon_int], color)
            annotated = cv2.addWeighted(
                overlay, OVERLAY_FILL_ALPHA, annotated, 1 - OVERLAY_FILL_ALPHA, 0
            )
            for polygon_int, candidate_type in polygons_px:
                color = OVERLAY_COLORS.get(candidate_type, (255, 255, 255))
                cv2.polylines(annotated, [polygon_int], isClosed=True, color=color, thickness=2)

        success, buffer = cv2.imencode(".png", annotated)
        if not success:
            return ""
        return base64.b64encode(buffer).decode("ascii")

    @staticmethod
    def _real_area(pixel_area: float, width: int, height: int, extent: Extent3857) -> float:
        meters_per_pixel_x = (extent.max_x - extent.min_x) / width
        meters_per_pixel_y = (extent.max_y - extent.min_y) / height
        return pixel_area * meters_per_pixel_x * meters_per_pixel_y

    @staticmethod
    def _calculate_shape_features(polygon: np.ndarray) -> dict[str, float | str]:
        """Calculate the shape quality used to estimate panel placement.

        The metrics follow the supplied notebook, but OpenCV's convex hull is
        used so the API does not need an additional SciPy dependency.
        """
        polygon_float = polygon.astype(np.float32)
        polygon_area = float(cv2.contourArea(polygon_float))
        hull = cv2.convexHull(polygon_float)
        hull_area = float(cv2.contourArea(hull))
        solidity = polygon_area / hull_area if hull_area > 0 else 0.0

        x, y, width, height = cv2.boundingRect(polygon.astype(np.int32))
        del x, y  # Only dimensions are required for the shape metrics.
        bbox_area = width * height
        fill_ratio = polygon_area / bbox_area if bbox_area > 0 else 0.0
        aspect_ratio = max(width, height) / max(1, min(width, height))

        if aspect_ratio < 1.5:
            aspect_score = 1.0
        elif aspect_ratio < 2.5:
            aspect_score = 0.9
        elif aspect_ratio < 4.0:
            aspect_score = 0.8
        else:
            aspect_score = 0.7

        shape_score = solidity * 0.5 + fill_ratio * 0.3 + aspect_score * 0.2
        if shape_score >= 0.90:
            shape_grade, shape_efficiency = "A", 0.95
        elif shape_score >= 0.80:
            shape_grade, shape_efficiency = "B", 0.90
        elif shape_score >= 0.70:
            shape_grade, shape_efficiency = "C", 0.85
        elif shape_score >= 0.60:
            shape_grade, shape_efficiency = "D", 0.80
        else:
            shape_grade, shape_efficiency = "E", 0.70

        return {
            "shape_score": round(shape_score, 3),
            "shape_grade": shape_grade,
            "shape_efficiency": shape_efficiency,
            "recommended_layout": "Landscape" if width >= height else "Portrait",
        }

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

    @staticmethod
    def _to_geometry(points: np.ndarray | list[list[float]]) -> BaseGeometry | None:
        if len(points) < 3:
            return None
        geometry = Polygon(points)
        if not geometry.is_valid:
            geometry = geometry.buffer(0)
        return None if geometry.is_empty else geometry

    @staticmethod
    def _round_or_none(value: float | None) -> float | None:
        return None if value is None else round(value, 2)

    @staticmethod
    def _overlaps_watermark(polygon_int: np.ndarray, width: int, height: int) -> bool:
        """True if most of the polygon's bounding box sits in the fixed
        bottom-left corner where VWorld stamps its "V-WORLD" attribution logo.
        """
        box_x, box_y, box_w, box_h = cv2.boundingRect(polygon_int)
        if box_w <= 0 or box_h <= 0:
            return False

        watermark_x_max = width * WATERMARK_WIDTH_FRACTION
        watermark_y_min = height * (1 - WATERMARK_HEIGHT_FRACTION)

        overlap_x = min(box_x + box_w, watermark_x_max) - max(box_x, 0.0)
        overlap_y = min(box_y + box_h, float(height)) - max(box_y, watermark_y_min)
        overlap_area = max(0.0, overlap_x) * max(0.0, overlap_y)

        return (overlap_area / (box_w * box_h)) > WATERMARK_OVERLAP_THRESHOLD
