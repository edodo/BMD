"""FastAPI 애플리케이션 진입점.

다른 프로젝트에 통합 시: create_app()을 import 하여 마운트하거나,
api_router를 상위 앱에 prefix를 붙여 include 하면 된다.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.v1.router import api_router
from app.core.config import settings
from app.db.session import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.dicom_dir.mkdir(parents=True, exist_ok=True)
    settings.derived_dir.mkdir(parents=True, exist_ok=True)
    await init_db()
    yield


def create_app() -> FastAPI:
    # StaticFiles가 마운트 시점에 디렉터리 존재를 확인하므로 lifespan보다 먼저 생성해야 한다
    # (신규/빈 바인드 마운트로 처음 뜰 때 storage/derived가 아직 없어 죽는 문제 방지).
    settings.dicom_dir.mkdir(parents=True, exist_ok=True)
    settings.derived_dir.mkdir(parents=True, exist_ok=True)

    app = FastAPI(
        title=settings.app_name,
        lifespan=lifespan,
        debug=settings.debug,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(api_router, prefix=settings.api_v1_prefix)

    # 파생 산출물(preview/gradcam/mask) 정적 서빙
    app.mount(
        "/derived",
        StaticFiles(directory=str(settings.derived_dir)),
        name="derived",
    )

    @app.get("/health", tags=["health"])
    async def health():
        return {"status": "ok", "app": settings.app_name}

    return app


app = create_app()
