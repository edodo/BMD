"""정밀도(LSC) 보정 엔드포인트 — 재위치 섭동 재측정으로 RMS-CV%/LSC% 산출.

'유의미한 변화'의 기준(config.significantLossPct 하드코딩 10%)을 실제로 측정된
값으로 바꾸기 위한 엔드포인트. 노트북 v18의 D0(precision_lsc)와 동일 산식이며,
합성 데모 대신 이 배포에 실제 업로드된 스터디로 계산한다.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_doctor
from app.core.config import settings
from app.db.session import get_db
from app.models import Doctor, PrecisionCalibration, StudyStatus, XrayStudy
from app.schemas import PrecisionCalibrationOut
from app.services.inference.engine import get_engine

router = APIRouter()

# 최근 완료 스터디 중 이만큼만 표본으로 사용 (매번 전체 이력을 재측정하면
# 스터디당 n_repeats회 추론이 들어가 느려짐 -- 보정은 자주 돌리는 작업이 아님).
CALIBRATION_SAMPLE_SIZE = 8


@router.get("/lsc", response_model=PrecisionCalibrationOut | None)
async def get_latest_calibration(
    doctor: Doctor = Depends(get_current_doctor),
    db: AsyncSession = Depends(get_db),
):
    """가장 최근 정밀도 보정 결과. 한 번도 보정하지 않았으면 null
    (프론트는 이 경우 '아직 보정되지 않음'을 명시해야 한다 — 조용히 임의의
    기본값으로 대체하면 노트북에서와 같은 '미검증 임계값' 문제가 되풀이된다).
    """
    row = (
        await db.execute(
            select(PrecisionCalibration)
            .order_by(PrecisionCalibration.computed_at.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    return row


@router.post("/calibrate", response_model=PrecisionCalibrationOut)
async def calibrate(
    doctor: Doctor = Depends(get_current_doctor),
    db: AsyncSession = Depends(get_db),
):
    """완료된 스터디 중 최근 N개를 재위치 섭동으로 재측정해 LSC%를 재산출한다.

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

    studies = (
        (
            await db.execute(
                select(XrayStudy)
                .where(XrayStudy.status == StudyStatus.COMPLETED)
                .order_by(XrayStudy.uploaded_at.desc())
                .limit(CALIBRATION_SAMPLE_SIZE)
            )
        )
        .scalars()
        .all()
    )
    if len(studies) < 2:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least 2 completed studies to calibrate precision "
            f"(have {len(studies)}). Upload more studies first.",
        )

    dicom_paths = [settings.dicom_dir / s.dicom_path for s in studies]
    result = engine.measure_precision(dicom_paths)
    if result is None:
        raise HTTPException(
            status_code=422,
            detail="Precision calibration failed on every sampled study "
            "(none produced 2+ usable repeat measurements).",
        )

    row = PrecisionCalibration(
        n_studies=result["n_studies"],
        n_repeats=result["n_repeats"],
        rms_cv_pct=result["rms_cv_pct"],
        lsc_pct=result["lsc_pct"],
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return row
