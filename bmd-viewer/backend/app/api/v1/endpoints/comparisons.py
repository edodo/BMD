"""저장된 비교(북마크) + 다중 환자 BMD 비교 엔드포인트."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_doctor
from app.core.config import settings
from app.db.session import get_db
from app.models import (
    BmdMeasurement,
    ComparisonType,
    Doctor,
    Patient,
    SavedComparison,
    XrayStudy,
)
from app.schemas import (
    BmdTrendPoint,
    ComparisonCreate,
    ComparisonOut,
    MultiPatientBmdOut,
)
from app.services.density_diff_render import render_density_diff_png

router = APIRouter()


@router.get("/density-diff.png")
async def density_diff_png(
    pre_study_id: str,
    post_study_id: str,
    db: AsyncSession = Depends(get_db),
):
    """두 스터디의 L4 밀도 격자 차이를 matplotlib으로 렌더링한 PNG.

    <img src>에서 바로 쓸 수 있도록 인증 없이 제공한다 -- 브라우저 img 태그는
    Authorization 헤더를 못 붙이므로, 기존 /derived 정적 파일(L4 크롭/히트맵)과
    같은 보안 수준(스터디 UUID를 모르면 접근 불가)으로 맞춘다.
    pre/post 두 스터디가 서로 다른 시점의 격자 해상도(bmd_diff_grid_n 변경 등)를
    가져도 render_density_diff_png가 상대 위치로 정렬하므로 그대로 비교 가능.
    """
    pre = await db.scalar(
        select(BmdMeasurement).where(BmdMeasurement.study_id == pre_study_id)
    )
    post = await db.scalar(
        select(BmdMeasurement).where(BmdMeasurement.study_id == post_study_id)
    )
    if not pre or not pre.l4_density_grid:
        raise HTTPException(status_code=404, detail="pre study has no density grid")
    if not post or not post.l4_density_grid:
        raise HTTPException(status_code=404, detail="post study has no density grid")

    png = render_density_diff_png(
        pre.l4_density_grid,
        post.l4_density_grid,
        post.l4_density_grid_aspect,
        pre_bar=pre.bmd_value,
        post_bar=post.bmd_value,
    )
    return Response(
        content=png,
        media_type="image/png",
        headers={"Cache-Control": "no-store"},
    )


@router.post("", response_model=ComparisonOut, status_code=201)
async def create_comparison(
    payload: ComparisonCreate,
    doctor: Doctor = Depends(get_current_doctor),
    db: AsyncSession = Depends(get_db),
):
    """비교 북마크 저장 — 계산 결과가 아니라 '무엇을 비교했는지'만 저장한다.

    다시 열 때는 study/trend 엔드포인트로 최신 데이터를 다시 조회하므로
    원본 데이터(재추론, 삭제 등)와 어긋날 수 없다.
    """
    if payload.type == ComparisonType.PRE_POST and not payload.patient_id:
        raise HTTPException(
            status_code=400, detail="patient_id is required for pre_post comparisons."
        )
    if payload.type == ComparisonType.MULTI_PATIENT and not payload.patient_ids:
        raise HTTPException(
            status_code=400,
            detail="patient_ids is required for multi_patient comparisons.",
        )
    if payload.patient_id:
        patient = await db.get(Patient, payload.patient_id)
        if not patient or patient.doctor_id != doctor.id:
            raise HTTPException(status_code=404, detail="Patient not found")

    comparison = SavedComparison(
        doctor_id=doctor.id,
        type=payload.type,
        title=payload.title,
        patient_id=payload.patient_id,
        pre_study_id=payload.pre_study_id,
        post_study_ids=payload.post_study_ids,
        patient_ids=payload.patient_ids,
        config=payload.config,
    )
    db.add(comparison)
    await db.commit()
    await db.refresh(comparison)
    return comparison


@router.get("", response_model=list[ComparisonOut])
async def list_comparisons(
    patient_id: str | None = None,
    type: ComparisonType | None = None,
    doctor: Doctor = Depends(get_current_doctor),
    db: AsyncSession = Depends(get_db),
):
    """저장된 비교 목록 (현재 의사 소유분만, 최신순)."""
    stmt = select(SavedComparison).where(SavedComparison.doctor_id == doctor.id)
    if patient_id:
        stmt = stmt.where(SavedComparison.patient_id == patient_id)
    if type:
        stmt = stmt.where(SavedComparison.type == type)
    stmt = stmt.order_by(SavedComparison.created_at.desc())
    return (await db.scalars(stmt)).all()


@router.get("/multi-patient-bmd", response_model=list[MultiPatientBmdOut])
async def multi_patient_bmd(
    patient_ids: str = Query(..., description="쉼표로 구분된 환자 ID 목록"),
    doctor: Doctor = Depends(get_current_doctor),
    db: AsyncSession = Depends(get_db),
):
    """여러 환자의 BMD 추세를 한 번에 — trends.py의 단일 환자 조회를 그대로
    반복한다. 현재 의사 소유가 아니거나 존재하지 않는 ID는 조용히 건너뛴다
    (하나 잘못된 ID 때문에 나머지 정상 요청까지 막지 않기 위함)."""
    ids = [pid.strip() for pid in patient_ids.split(",") if pid.strip()]
    out: list[MultiPatientBmdOut] = []
    for pid in ids:
        patient = await db.get(Patient, pid)
        if not patient or patient.doctor_id != doctor.id:
            continue
        # trends.py와 동일한 이유로 computed_at이 아니라 acquired_at(없으면
        # uploaded_at)을 써야 History에서 고친 날짜가 그래프에도 반영된다.
        measured_key = func.coalesce(XrayStudy.acquired_at, XrayStudy.uploaded_at)
        rows = (
            await db.execute(
                select(
                    XrayStudy.id,
                    measured_key,
                    BmdMeasurement.bmd_value,
                    BmdMeasurement.t_score,
                )
                .join(BmdMeasurement, BmdMeasurement.study_id == XrayStudy.id)
                .where(XrayStudy.patient_id == pid)
                .order_by(measured_key.asc())
            )
        ).all()
        points = [
            BmdTrendPoint(
                study_id=r[0], measured_at=r[1], bmd_value=r[2], t_score=r[3]
            )
            for r in rows
        ]
        delta_pct = None
        if len(points) >= 2 and points[0].bmd_value:
            delta_pct = round(
                (points[-1].bmd_value - points[0].bmd_value)
                / points[0].bmd_value
                * 100,
                2,
            )
        out.append(
            MultiPatientBmdOut(
                patient_id=pid,
                patient_name=patient.full_name,
                target_vertebra=settings.target_vertebra,
                points=points,
                delta_percent=delta_pct,
            )
        )
    return out


@router.get("/{comparison_id}", response_model=ComparisonOut)
async def get_comparison(
    comparison_id: str,
    doctor: Doctor = Depends(get_current_doctor),
    db: AsyncSession = Depends(get_db),
):
    comparison = await db.get(SavedComparison, comparison_id)
    if not comparison or comparison.doctor_id != doctor.id:
        raise HTTPException(status_code=404, detail="Comparison not found")
    return comparison


@router.delete("/{comparison_id}", status_code=204)
async def delete_comparison(
    comparison_id: str,
    doctor: Doctor = Depends(get_current_doctor),
    db: AsyncSession = Depends(get_db),
):
    comparison = await db.get(SavedComparison, comparison_id)
    if not comparison or comparison.doctor_id != doctor.id:
        raise HTTPException(status_code=404, detail="Comparison not found")
    await db.delete(comparison)
    await db.commit()
    return None
