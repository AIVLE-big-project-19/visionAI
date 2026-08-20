# Vision AI Server

## 1. 프로젝트 개요

Vision AI Server는 **YOLOv8 Segmentation 모델과 후보지 GeoPackage(GPKG)를 이용하여 위성(항공) 이미지에서 태양광 설치 가능 영역을 분석하는 AI 서버**입니다.

AI Server는 이미지 분석과 후보지 polygon 기반 공간 연산을 담당하며, 주소 검색이나 데이터베이스 저장은 수행하지 않습니다.

프로젝트 구조는 다음과 같습니다.

```
React
    │
    ▼
Spring Boot (Backend)
    │
    ▼
Vision AI Server (FastAPI)
    │
    ├── Candidate Parcels GPKG
    │
    ▼
YOLOv8 Segmentation
```

각 역할은 다음과 같습니다.

- **Vision AI Server**
  - 전달받은 이미지와 `extent3857`을 기준으로 후보지 선택
  - 전달받은 이미지를 YOLO 모델로 분석
  - 후보지 polygon, 패널 배치, 거리 및 면적 계산
  - 최종 시각화 이미지와 분석 결과를 JSON으로 반환

---

# 2. 동작 방식

전체 처리 과정은 다음과 같습니다.

```
주소 입력

↓

Backend에서 후보지 조회

↓

VWorld Static Map API 호출

↓

위성 이미지(PNG/JPG) 및 extent3857 획득

↓

Vision AI Server 호출

↓

extent3857과 교차하는 후보지 polygon 선택

↓

YOLO 추론 및 패널 배치

↓

거리 · 면적 · 최종 시각화 이미지 생성

↓

JSON 반환

↓

Backend에서 DB 저장
```

Vision AI Server는 **이미지 1장당 후보지 1건을 선택하여 1회 추론**하도록 설계되어 있습니다.

후보지가 여러 개인 경우 Backend에서 각각의 이미지에 대해 AI Server를 여러 번 호출합니다.

### 후보지 데이터

`data/candidate_parcels.gpkg`에는 다음 두 종류의 후보지가 함께 저장되어 있습니다.

| candidate_type | 설명 |
|---|---|
| land | 토지(유휴부지) polygon |
| building | 건물 옥상 polygon |

GPKG의 geometry는 EPSG:5179로 저장됩니다. 서버는 시작 시 후보지 geometry를 EPSG:3857로 변환하여, Backend가 전달한 `extent3857`과 후보지 선택·거리 계산·pixel 변환을 수행합니다.

건물 후보는 VWorld building polygon과 항공영상 간 위치 오차를 고려하여, building mask 비교 시에만 5m spatial tolerance를 적용합니다. 토지 후보의 공간 연산은 기존 방식 그대로 수행합니다.

---

# 3. 실행 방법

필요 패키지를 설치한 후 환경 변수를 설정하고 서버를 실행합니다.

```powershell
pip install -r requirements.txt

$env:MODEL_PATH="C:\models\bestv2.pt"
$env:MODEL_VERSION="solar-yolov8-seg-v1"

uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 환경 변수

| 변수 | 설명 | 기본값 |
|------|------|------|
| MODEL_PATH | YOLO(.pt) 모델 파일 경로 | 필수 |
| MODEL_VERSION | 모델 버전 정보 | 모델 파일명 |
| MIN_CONFIDENCE | YOLO 최소 신뢰도 | 0.50 |
| MIN_PIXEL_AREA | 분석할 후보지의 최소 pixel 면적 | 700 |
| DEBUG | `true`이면 원본 크기 최종 시각화 PNG를 `debug` 폴더에 저장 | false |

모델과 GPKG 후보지 데이터는 서버가 시작될 때 **한 번만 로드**되며, 요청마다 다시 로드하지 않습니다.

---

# 4. API

## POST /predict

Vision AI Server는 하나의 API를 제공합니다.

### Request

Content-Type

```
multipart/form-data
```

### 전달해야 하는 값

| 파라미터 | 설명 |
|----------|------|
| image | 분석할 위성(항공) 이미지(PNG/JPG) |
| extent3857 | 해당 이미지가 나타내는 실제 지도 영역(EPSG:3857 좌표) |

### image

Backend에서 VWorld Static Map API를 통해 받은 이미지를 그대로 전달합니다.

이미지는 디스크에 저장하지 않고 메모리에서 바로 분석합니다.

### extent3857

이미지가 실제 지도에서 어느 영역을 나타내는지 알려주는 좌표 정보입니다.

후보지 선택, 후보지 polygon의 pixel 변환, 거리 계산에 사용되므로 반드시 필요합니다.

다음 형식 모두 사용할 수 있습니다.

JSON Object

```json
{
  "minX": 14135000.0,
  "minY": 4518000.0,
  "maxX": 14136000.0,
  "maxY": 4519000.0
}
```

JSON Array

```json
[
  14135000.0,
  4518000.0,
  14136000.0,
  4519000.0
]
```

Comma-separated String

```
14135000.0,4518000.0,14136000.0,4519000.0
```

---

# 5. Response

AI Server는 후보지 1건의 분석 결과와 최종 시각화 이미지를 JSON object로 반환합니다.

```json
{
  "predictions": [
    {
      "candidate_type": "land",
      "detected_type": "land",
      "confidence": 0.9231,
      "polygon": [[14135200.5, 4518750.2]],
      "pixel_area": 1223.5,
      "real_area": 186.69,
      "distance_to_road_px": 18.25,
      "distance_to_building_px": 42.1,
      "distance_to_road_m": 5.42,
      "distance_to_building_m": 12.61,
      "shape_score": 81.54,
      "shape_grade": "B",
      "shape_efficiency": 0.815,
      "recommended_layout": "Landscape",
      "usable_area": 152.15,
      "estimated_panel_count": 54,
      "candidate_id": "land:SOLAR_01485:0",
      "pnu": "...",
      "address": "...",
      "panel_layout": [],
      "valid_panel_count": 48,
      "removed_panel_count": 6,
      "installed_area": 134.17,
      "model_version": "solar-yolov8-seg-v1"
    }
  ],
  "final_visualization_image": "Base64-encoded JPEG string"
}
```

### 반환 값 설명

| 항목 | 설명 |
|------|------|
| predictions | 후보지 분석 결과 배열 |
| candidate_type | GPKG 후보지 종류: `land` 또는 `building` |
| detected_type | 후보지 영역과 가장 많이 겹치는 YOLO 탐지 클래스 |
| confidence | `detected_type`의 YOLO 예측 신뢰도 |
| polygon | 선택된 후보지 polygon 좌표(EPSG:3857) |
| pixel_area | 이미지 상의 후보지 pixel 면적 |
| real_area | GPKG의 후보지 면적(m²) |
| distance_to_road_px | 후보지와 YOLO road mask의 최단 거리(pixel) |
| distance_to_building_px | 후보지와 YOLO building mask의 최단 거리(pixel) |
| distance_to_road_m | 후보지와 YOLO road mask의 최단 거리(m) |
| distance_to_building_m | 후보지와 YOLO building mask의 최단 거리(m) |
| shape_score | polygon 형상 기반 패널 배치 적합도 점수 |
| shape_grade | polygon 형상 등급(A~D) |
| shape_efficiency | polygon 형상 기반 사용 가능 면적 비율 |
| recommended_layout | 권장 패널 방향: `Landscape` 또는 `Portrait` |
| usable_area | 형상 효율을 적용한 사용 가능 면적(m²) |
| estimated_panel_count | 사용 가능 면적 기준 예상 패널 수 |
| candidate_id | GPKG 후보지 고유 ID |
| pnu | 토지 후보의 PNU. 건물 후보는 `null` |
| address | 후보지 주소 |
| panel_layout | 패널별 위치·크기·유효 여부 정보 |
| valid_panel_count | 유효 패널 수 |
| removed_panel_count | road/building 거리 조건으로 제외된 패널 수 |
| installed_area | 유효 패널 총 면적(m²) |
| model_version | 사용한 모델 버전 |
| final_visualization_image | 유효 패널과 후보지 경계가 표시된 480×408 JPEG의 Base64 문자열 |

후보지가 선택되지 않거나 후보지 pixel 면적이 최소값보다 작으면 다음과 같이 반환합니다. 이 경우 `final_visualization_image`에는 축소된 원본 이미지가 포함됩니다.

```json
{
  "predictions": [],
  "final_visualization_image": "Base64-encoded JPEG string"
}
```

---

# 6. 현재 적용된 탐지 조건

현재 AI Server는 다음 조건을 사용합니다.

- Confidence ≥ 0.50
- Pixel Area ≥ 700
- 대상 클래스
  - building
  - parking_lot
  - land
- 패널 크기: 2.465m × 1.134m
- 패널 간격: 0.2m
- 후보지 경계 여백: 0.4m
- road 이격 기준: 20 pixel
- building 이격 기준: 10 pixel

---

# 7. 역할 범위

Vision AI Server는 **후보지 기반 AI 추론과 결과 시각화만 담당**합니다.

수행하는 기능

- 후보지 GPKG 로드 및 extent 기반 후보지 선택
- 이미지 분석 및 YOLO 추론
- 후보지 polygon의 pixel 변환
- 패널 배치 및 거리 계산
- 실제 면적 및 사용 가능 면적 계산
- 최종 시각화 JPEG 생성
- JSON 반환

수행하지 않는 기능

- 주소 검색
- 후보지 원천 데이터 생성·수정
- VWorld API 호출
- 데이터베이스 저장
- 경제성 분석
- 적합도 종합 계산
- LLM 응답 생성

이 기능들은 모두 Backend(Spring Boot)에서 담당합니다.
