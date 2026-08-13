"""데이터베이스 연결 및 세션 관리.

비동기 SQLAlchemy를 사용한다. SQLite가 기본이며,
database_url만 교체하면 PostgreSQL 등으로 무수정 전환 가능하도록 구성한다.
"""
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.core.config import settings


class Base(DeclarativeBase):
    """모든 ORM 모델의 베이스 클래스."""


# SQLite는 동시 쓰기에 제약이 있으므로 connect_args로 보완
connect_args = (
    {"check_same_thread": False}
    if settings.database_url.startswith("sqlite")
    else {}
)

engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,
    connect_args=connect_args,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI 의존성으로 주입할 DB 세션."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db() -> None:
    """앱 시작 시 테이블 생성 (개발/SQLite용). 운영은 Alembic 권장."""
    # 모델을 등록하기 위해 import
    from app import models  # noqa: F401

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(_apply_lightweight_migrations)


def _apply_lightweight_migrations(conn) -> None:
    """create_all이 추가하지 못하는 신규 컬럼을 보충한다 (개발/SQLite용).

    SQLite의 `CREATE TABLE`은 이미 존재하는 테이블에 컬럼을 더하지 않으므로,
    기존 DB를 삭제하지 않고도 신규 컬럼을 사용할 수 있게 idempotent하게 처리.
    운영 환경에서는 Alembic 마이그레이션으로 대체할 것.
    """
    from sqlalchemy import inspect, text

    inspector = inspect(conn)

    def _add_missing(table: str, additions: dict[str, str]) -> None:
        try:
            existing = {c["name"] for c in inspector.get_columns(table)}
        except Exception:  # noqa: BLE001 — 테이블이 없으면 create_all이 처리
            return
        for col, ddl in additions.items():
            if col not in existing:
                conn.execute(
                    text(f"ALTER TABLE {table} ADD COLUMN {col} {ddl}")
                )

    # (컬럼명, DDL 타입) — 누락 시에만 ADD COLUMN
    _add_missing(
        "bmd_measurements",
        {
            "xai_overlay_path": "VARCHAR(512)",
            "xai_l4_cam_path": "VARCHAR(512)",
            "xai_bar_l1l5_path": "VARCHAR(512)",
            "density_low": "FLOAT",
            "density_high": "FLOAT",
            "roi_mean_intensity": "FLOAT",
            "reliable": "BOOLEAN DEFAULT 1",
            "reliability_warning": "TEXT",
            "l4_density_grid": "JSON",
        },
    )
    _add_missing(
        "xray_studies",
        {
            "overlay_path": "VARCHAR(512)",
            "view_position": "VARCHAR(8)",
            "view_source": "VARCHAR(8)",
        },
    )
    _add_missing(
        "vertebra_segments",
        {
            "bar": "FLOAT",
            "qc_status": "VARCHAR(8)",
            "qc_message": "TEXT",
            "glcm_contrast": "FLOAT",
            "glcm_correlation": "FLOAT",
            "glcm_energy": "FLOAT",
            "glcm_homogeneity": "FLOAT",
            "glcm_entropy": "FLOAT",
            "vario_slope": "FLOAT",
            "fractal_dim": "FLOAT",
            "anomaly_mse": "FLOAT",
            "anomaly_pct": "FLOAT",
            "anomaly_shap": "JSON",
            "bar_z": "FLOAT",
            "bhi_z": "FLOAT",
            "bhi_pct": "FLOAT",
            "category": "VARCHAR(64)",
        },
    )
