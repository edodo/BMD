"""애플리케이션 설정.

모든 설정은 환경변수로 주입 가능하도록 구성한다.
다른 프로젝트에 통합할 때 prefix(BMD_)만 바꾸거나 .env를 교체하면 된다.
"""
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="BMD_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- 일반 ---
    app_name: str = "BMD Viewer API"
    api_v1_prefix: str = "/api/v1"
    debug: bool = False

    # --- 데이터베이스 ---
    # SQLite 기본. 다른 프로젝트 통합 시 PostgreSQL URL로 교체 가능.
    database_url: str = Field(
        default=f"sqlite+aiosqlite:///{BASE_DIR}/storage/bmd.db"
    )

    # --- 인증 (JWT) ---
    secret_key: str = "CHANGE_ME_IN_PRODUCTION"
    access_token_expire_minutes: int = 480  # 8시간 (의료 근무 1교대)
    algorithm: str = "HS256"

    # --- 스토리지 ---
    storage_dir: Path = BASE_DIR / "storage"
    dicom_dir: Path = BASE_DIR / "storage" / "dicom"
    derived_dir: Path = BASE_DIR / "storage" / "derived"

    # --- ML / 추론 ---
    ml_model_dir: Path = BASE_DIR / "ml_models"
    inference_engine: str = "yolo"  # "yolo" | "stub"
    inference_device: str = "cpu"  # "cuda" 가능 (RTX 5060 / sm_120)
    # L4를 BMD 계산 기준 척추로 사용 (요구사항)
    target_vertebra: str = "L4"

    # --- CORS ---
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    # --- 비동기 작업 ---
    # 단일 노드는 in-process BackgroundTasks, 확장 시 celery/redis로 교체
    worker_backend: str = "background"  # "background" | "celery"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
