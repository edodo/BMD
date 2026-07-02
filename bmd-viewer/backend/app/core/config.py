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
    # 분할 모델 가중치 파일명 (ml_model_dir 기준).
    # YOLO26m-seg L1~L5 (AP+LA 학습, imgsz=640).
    # .pt 사용 → Eigen-CAM(모델 주목도 XAI)까지 전 기능 지원.
    # .onnx(l1_l5_yolo26m_seg.onnx)로 바꾸면 onnxruntime 추론은 되지만
    #   torch forward hook이 없어 Eigen-CAM 히트맵은 생략된다.
    ml_model_file: str = "l1_l5_yolo26m_seg.pt"
    inference_engine: str = "yolo"  # "yolo" | "stub"
    inference_device: str = "cpu"  # "cuda" 가능 (RTX 5060 / sm_120)
    # L4를 BMD 계산 기준 척추로 사용 (요구사항)
    target_vertebra: str = "L4"

    # --- BMD 정규화 (유효 조직만 사용) ---
    # 고관절 수술 환자 영상은 (1) 촬영 여백/공기 배경, (2) 금속 인공관절이
    # 섞여 전체 percentile 기준을 오염시킨다. 아래 임계로 두 가지를 제외한다.
    #   배경  : Otsu 임계 이하 (거의 균일한 저밀도) → 연부조직 기준선에서 제외
    #   금속  : 유효 뼈 기준값(p95) × metal_ratio 이상 → 초고감쇠 인공물 제외
    bmd_exclude_background: bool = True
    # 금속 판정 배수. 뼈 p95의 이 배수를 넘으면 금속/인공물로 보고 제외.
    # 금속 없는 영상에선 max가 이 값을 넘지 않아 아무것도 제외되지 않는다.
    # 실제 인공관절 영상으로 보정 권장 (1.2~1.5 범위).
    bmd_metal_ratio: float = 1.3
    # L4 해면골(cancellous) ROI 크기. L4 분할 폴리곤을 중심으로 이 비율만큼
    # 축소해 중앙 해면골만 잡는다. 1.0=L4 전체(피질골·종판 포함),
    # 0.7=좁은 중심, 0.85=L4에 딱 맞으면서 피질골 edge는 회피(모델 소스와 동일).
    bmd_roi_shrink: float = 0.85

    # --- CORS ---
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    # --- 비동기 작업 ---
    # 단일 노드는 in-process BackgroundTasks, 확장 시 celery/redis로 교체
    worker_backend: str = "background"  # "background" | "celery"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
