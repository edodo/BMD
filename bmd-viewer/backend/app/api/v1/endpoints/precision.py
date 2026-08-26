"""정밀도(LSC) 보정 엔드포인트 — 재위치 섭동 재측정으로 RMS-CV%/LSC% 산출.

'유의미한 변화'의 기준(config.significantLossPct 하드코딩 10%)을 실제로 측정된
값으로 바꾸기 위한 엔드포인트. 노트북 v18의 D0(precision_lsc)와 동일 산식이며,
합성 데모 대신 이 배포에 실제 업로드된 스터디로 계산한다.

실제 계산(표본 조회 + 재측정 + 저장)은 app/services/precision_calibration.py에
공유돼 있다 -- 스터디가 완료될 때마다 자동으로도 같은 로직이 돌기 때문
(inference_worker.schedule_auto_calibration). 이 엔드포인트는 그 자동 주기를
기다리지 않고 지금 당장 강제로 재계산하고 싶을 때 쓴다.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_doctor
from app.db.session import get_db
from app.models import Doctor
from app.schemas import PrecisionCalibrationOut
from app.services.inference.engine import get_engine
from app.services.precision_calibration import (
    get_latest_calibration,
    run_calibration,
)

router = APIRouter()


@router.get("/lsc", response_model=PrecisionCalibrationOut | None)
async def get_lsc(
    doctor: Doctor = Depends(get_current_doctor),
    db: AsyncSession = Depends(get_db),
):
    """가장 최근 정밀도 보정 결과. 한 번도 보정하지 않았으면 null
    (프론트는 이 경우 '아직 보정되지 않음'을 명시해야 한다 — 조용히 임의의
    기본값으로 대체하면 노트북에서와 같은 '미검증 임계값' 문제가 되풀이된다).
    """
    return await get_latest_calibration(db)


@router.post("/calibrate", response_model=PrecisionCalibrationOut)
async def calibrate(
    doctor: Doctor = Depends(get_current_doctor),
    db: AsyncSession = Depends(get_db),
):
    """완료된 스터디 전체(최대 안전 상한까지)를 재위치 섭동으로 재측정해
    LSC%를 지금 즉시 재산출한다.

    doctor/patient에 종속되지 않는 시스템 속성이므로, 호출한 의사가 아니라
    전체 완료 스터디에서 표본을 뽑는다. 실제 YOLO 엔진이 필요하므로
    BMD_INFERENCE_ENGINE=stub일 때는 사용할 수 없다.
    """
    engine = get_engine()
    if not hasattr(engine, "measure_precision"):
        raise HTTPException(
            status_code=503,
            detail="Precision calibration requires the real YOLO inference "
            "engine (unavailable with BMD_INFERENCE_ENGINE=stub).",
        )

    row = await run_calibration(db)
    if row is None:
        raise HTTPException(
            status_code=400,
            detail="Need at least 2 completed studies to calibrate precision, "
            "or precision calibration failed on every sampled study. "
            "Upload more studies first.",
        )
    return row
