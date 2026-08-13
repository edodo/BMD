"""BMD 추세 엔드포인트 - 종단 변화량 차트 데이터."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_doctor
from app.core.config import settings
from app.db.session import get_db
from app.models import BmdMeasurement, Doctor, Patient, PrecisionCalibration, XrayStudy
from app.schemas import BmdTrendOut, BmdTrendPoint

router = APIRouter()


@router.get("/patients/{patient_id}/bmd-trend", response_model=BmdTrendOut)
async def bmd_trend(
    patient_id: str,
    doctor: Doctor = Depends(get_current_doctor),
    db: AsyncSession = Depends(get_db),
):
    """환자의 L4 BMD 시계열과 변화량 요약."""
    patient = await db.get(Patient, patient_id)
    if not patient or patient.doctor_id != doctor.id:
        raise HTTPException(status_code=404, detail="Patient not found")

    rows = (
        await db.execute(
            select(
                XrayStudy.id,
                BmdMeasurement.computed_at,
                BmdMeasurement.bmd_value,
                BmdMeasurement.t_score,
            )
            .join(BmdMeasurement, BmdMeasurement.study_id == XrayStudy.id)
            .where(XrayStudy.patient_id == patient_id)
            .order_by(BmdMeasurement.computed_at.asc())
        )
    ).all()

    points = [
        BmdTrendPoint(
            study_id=r[0], measured_at=r[1], bmd_value=r[2], t_score=r[3]
        )
        for r in rows
    ]

    delta_abs = delta_pct = None
    if len(points) >= 2:
        first, last = points[0].bmd_value, points[-1].bmd_value
        delta_abs = round(last - first, 4)
        if first:
            delta_pct = round((last - first) / first * 100, 2)

    # 측정 정밀도(LSC%) -- 보정 이력이 있으면 최신값을 함께 내려준다. 프론트가
    # 이 값과 delta_percent를 비교해 '재위치 노이즈 vs 실제 변화'를 판정한다.
    calibration = (
        await db.execute(
            select(PrecisionCalibration)
            .order_by(PrecisionCalibration.computed_at.desc())
            .limit(1)
        )
    ).scalar_one_or_none()

    return BmdTrendOut(
        patient_id=patient_id,
        target_vertebra=settings.target_vertebra,
        points=points,
        delta_absolute=delta_abs,
        delta_percent=delta_pct,
        lsc_percent=calibration.lsc_pct if calibration else None,
        lsc_calibrated_at=calibration.computed_at if calibration else None,
    )
