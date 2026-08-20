from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Annotated, Any

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, status
from pydantic import BaseModel, Field

from .config import get_settings
from .inference import Extent3857, YoloSegmentationService


class Prediction(BaseModel):
    candidate_type: str
    detected_type: str
    confidence: float = Field(ge=0, le=1)
    polygon: list[list[float]] = Field(description="EPSG: 3857 좌표계의 후보지 Polygon 꼭짓점 좌표입니다. [x, y]")
    pixel_area: float
    real_area: float = Field(description="GPKG 후보지 면적을 기준으로 계산한 실제 면적(m²)입니다.")
    distance_to_road_px: float | None = Field(
        description="후보지 polygon과 탐지된 도로 사이의 최단 거리(pixel)입니다."
    )
    distance_to_building_px: float | None = Field(
        description="후보지 polygon과 탐지된 건물 사이의 최단 거리(pixel)입니다."
    )
    distance_to_road_m: float | None = Field(
        description="후보지 polygon과 탐지된 도로 사이의 최단 거리(m)입니다."
    )
    distance_to_building_m: float | None = Field(
        description="후보지 polygon과 탐지된 건물 사이의 최단 거리(m)입니다."
    )
    shape_score: float = Field(description="Polygon 형상에 따른 패널 배치 적합도 점수입니다.")
    shape_grade: str = Field(description="형상 적합도 등급입니다. A가 가장 우수하고 D가 가장 낮습니다.")
    shape_efficiency: float = Field(
        description="형상 등급을 기준으로 계산한 사용 가능한 면적 효율입니다."
    )
    recommended_layout: str = Field(description="권장 패널 배치 방향입니다. Landscape 또는 Portrait입니다.")
    usable_area: float = Field(
        description="형상 및 배치 효율을 적용한 사용 가능 면적(m²)입니다."
    )
    estimated_panel_count: int = Field(
        description="사용 가능 면적에 설치 가능한 550W 패널의 예상 수량입니다."
    )
    candidate_id: str | None = None
    pnu: str | None = None
    address: str | None = None
    panel_layout: list[dict[str, Any]] | None = None
    valid_panel_count: int | None = None
    removed_panel_count: int | None = None
    installed_area: float | None = None
    model_version: str


class PredictResponse(BaseModel):
    predictions: list[Prediction]
    final_visualization_image: str = Field(
        description="유효 태양광 패널이 포함된 480X480 JPEG 최종 시각화 이미지의 Base64 문자열입니다."
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 모델은 요청마다 다시 불러오지 않고 FastAPI 프로세스 시작 시 한 번만 로드함
    app.state.predictor = YoloSegmentationService(get_settings())
    yield


app = FastAPI(title="Solar AI Inference API", version="1.0.0", lifespan=lifespan)


def parse_extent3857(value: str) -> Extent3857:
    """JSON 객체, JSON 배열 또는 쉼표로 구분된 minX, minY, maxX, maxY 형식을 처리합니다."""
    try:
        parsed: Any = json.loads(value)
    except json.JSONDecodeError:
        parsed = [part.strip() for part in value.split(",")]

    try:
        if isinstance(parsed, dict):
            extent = Extent3857(
                min_x=float(parsed["minX"]), min_y=float(parsed["minY"]),
                max_x=float(parsed["maxX"]), max_y=float(parsed["maxY"]),
            )
        elif isinstance(parsed, list) and len(parsed) == 4:
            extent = Extent3857(*(float(item) for item in parsed))
        else:
            raise ValueError
        extent.validate()
        return extent
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "extent3857은 {minX,minY,maxX,maxY}, "
                "[minX,minY,maxX,maxY], 또는 'minX,minY,maxX,maxY' 형식이어야 합니다."
            ),
        ) from exc


@app.post("/predict", response_model=PredictResponse)
async def predict(
    request: Request,
    image: Annotated[UploadFile, File(description="PNG/JPEG map image")],
    extent3857: Annotated[str, Form(description="EPSG:3857 좌표계의 지도 범위")],
) -> dict[str, Any]:

    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=415,
            detail="image는 이미지 파일이어야 합니다.",
        )

    raw_image = await image.read()

    try:
        decoded = cv2.imdecode(
            np.frombuffer(raw_image, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
    finally:
        await image.close()

    if decoded is None:
        raise HTTPException(
            status_code=422,
            detail="OpenCV에서 이미지를 디코딩할 수 없습니다.",
        )

    extent = parse_extent3857(extent3857)

    predictions, final_visualization_image = request.app.state.predictor.predict(
        decoded,
        extent,
    )

    return {
        "predictions": predictions,
        "final_visualization_image": final_visualization_image,
    }
