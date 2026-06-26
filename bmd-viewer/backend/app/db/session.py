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
