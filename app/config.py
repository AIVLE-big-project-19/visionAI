from dataclasses import dataclass
from pathlib import Path
import os

PROJECT_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    model_path: Path
    model_version: str
    min_confidence: float = 0.50
    min_pixel_area: float = 700.0


def get_settings() -> Settings:
    """애플리케이션 시작 시 설정을 한 번 읽어옵니다."""
    _load_dotenv(PROJECT_DIR / ".env")
    model_path_value = os.getenv("MODEL_PATH")
    if not model_path_value:
        raise RuntimeError("MODEL_PATH 환경 변수는 YOLO.pt 모델 파일을 가리켜야 합니다.")

    model_path = Path(model_path_value)
    # .env의 상대 MODEL_PATH는 Unicorn을 실행한 위치와 관계없이 프로젝트 디렉터리를 기준으로 처리함
    if not model_path.is_absolute():
        model_path = PROJECT_DIR / model_path

    return Settings(
        model_path=model_path.resolve(),
        model_version=os.getenv("MODEL_VERSION", model_path.stem),
        min_confidence=float(os.getenv("MIN_CONFIDENCE", "0.50")),
        min_pixel_area=float(os.getenv("MIN_PIXEL_AREA", "700")),
    )


def _load_dotenv(dotenv_path: Path) -> None:
    """추가 의존성 없이 간단한 KEY=VALUE 형식의 .env 파일을 불러옵니다.

    이미 설정된 프로세스 환경 변수가 우선이므로 배포 환경 설정으로 로컬 개발용 .env 값을 안전하게 덮어쓸 수 있습니다.
    """
    if not dotenv_path.is_file():
        return

    for line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)
