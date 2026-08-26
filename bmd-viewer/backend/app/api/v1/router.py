"""v1 API 라우터 집약."""
from fastapi import APIRouter

from app.api.v1.endpoints import (
    auth,
    comparisons,
    patients,
    precision,
    studies,
    trends,
)

api_router = APIRouter()
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(patients.router, prefix="/patients", tags=["patients"])
api_router.include_router(studies.router, tags=["studies"])
api_router.include_router(trends.router, tags=["trends"])
api_router.include_router(precision.router, prefix="/precision", tags=["precision"])
api_router.include_router(
    comparisons.router, prefix="/comparisons", tags=["comparisons"]
)
