# Vision AI Server

## 1. 프로젝트 개요

Vision AI Server는 **YOLOv8 Segmentation 모델을 이용하여 위성(항공) 이미지에서 설치 가능한 영역을 탐지하는 AI 서버**입니다.

AI Server는 **이미지 분석만 담당**하며, 주소 검색이나 데이터베이스 저장 등의 기능은 수행하지 않습니다.

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
    ▼
YOLOv8 Segmentation
```

각 역할은 다음과 같습니다.

- **Vision AI Server**
  - 전달받은 이미지를 YOLO 모델로 분석
  - 설치 가능 영역 탐지
  - Polygon 및 면적 계산
  - JSON 형태로 결과 반환

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

위성 이미지(PNG) 획득

↓

Vision AI Server 호출

↓

YOLO 추론

↓

Polygon 및 면적 계산

↓

JSON 반환

↓

Backend에서 DB 저장
```

Vision AI Server는 **이미지 1장당 1회 추론**하도록 설계되어 있습니다.

후보지가 여러 개인 경우 Backend에서 각각의 이미지에 대해 AI Server를 여러 번 호출합니다.

예)

```
후보지 A → AI 분석

후보지 B → AI 분석

후보지 C → AI 분석
```

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

| 변수 | 설명 |
|------|------|
| MODEL_PATH | YOLO(.pt) 모델 파일 경로 |
| MODEL_VERSION | 모델 버전 정보 |

모델은 서버가 시작될 때 **한 번만 로드**되며,
요청이 들어올 때마다 다시 로드하지 않습니다.

---

# 4. API

## POST /predict

Vision AI Server는 하나의 API만 제공합니다.

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

YOLO가 계산한 Pixel Area를 실제 면적(m²)으로 변환하기 위해 반드시 필요합니다.

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

AI Server는 설치 가능한 영역을 탐지하여 JSON 배열 형태로 반환합니다.

```json
[
  {
    "candidate_type": "land",
    "confidence": 0.9231,
    "polygon": [
      [14135200.5, 4518750.2]
    ],
    "pixel_area": 1223.5,
    "real_area": 186.69,
    "model_version": "solar-yolov8-seg-v1"
  }
]
```

### 반환 값 설명

| 항목 | 설명 |
|------|------|
| candidate_type | 탐지된 영역의 종류 |
| confidence | 예측 신뢰도 |
| polygon | 탐지된 영역의 Polygon 좌표(EPSG:3857) |
| pixel_area | 이미지 상의 픽셀 면적 |
| real_area | 실제 면적(m²) |
| model_version | 사용한 모델 버전 |

탐지된 영역이 없는 경우

```json
[]
```

을 반환합니다.

---

# 6. 현재 적용된 탐지 조건

현재 AI Server는 기존 학습 모델의 조건을 그대로 사용합니다.

- Confidence ≥ 0.50
- Pixel Area ≥ 700
- 대상 클래스
  - building
  - parking_lot
  - land

---

# 7. 역할 범위

Vision AI Server는 **AI 추론만 담당**합니다.

수행하는 기능

- 이미지 분석
- YOLO 추론
- Polygon 생성
- Pixel Area 계산
- 실제 면적 계산
- JSON 반환

수행하지 않는 기능

- 주소 검색
- 유휴부지 조회
- VWorld API 호출
- 데이터베이스 저장
- 경제성 분석
- 적합도 계산
- LLM 응답 생성

이 기능들은 모두 Backend(Spring Boot)에서 담당합니다.