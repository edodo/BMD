"""환자 엔드포인트 - 목록/검색/상세/생성."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_doctor
from app.core.config import settings
from app.db.session import get_db
from app.models import BmdMeasurement, Doctor, Patient, XrayStudy
from app.schemas import (
    PatientCreate,
    PatientListItem,
    PatientOut,
    PatientUpdate,
)

router = APIRouter()


@router.get("", response_model=list[PatientListItem])
async def list_patients(
    q: str | None = Query(None, description="이름 또는 MRN 검색어"),
    limit: int = Query(50, le=200),
    offset: int = 0,
    doctor: Doctor = Depends(get_current_doctor),
    db: AsyncSession = Depends(get_db),
):
    """환자 목록 + 검색. 각 항목에 스터디 수와 최신 BMD를 포함."""
    stmt = select(Patient).where(Patient.doctor_id == doctor.id)
    if q:
        like = f"%{q}%"
        stmt = stmt.where(
            or_(Patient.full_name.ilike(like),
                Patient.medical_record_no.ilike(like))
        )
    stmt = stmt.order_by(Patient.full_name).limit(limit).offset(offset)
    patients = (await db.scalars(stmt)).all()

    items: list[PatientListItem] = []
    for p in patients:
        count = await db.scalar(
            select(func.count(XrayStudy.id)).where(
                XrayStudy.patient_id == p.id
            )
        )
        latest_bmd = await db.scalar(
            select(BmdMeasurement.bmd_value)
            .join(XrayStudy, XrayStudy.id == BmdMeasurement.study_id)
            .where(XrayStudy.patient_id == p.id)
            .order_by(BmdMeasurement.computed_at.desc())
            .limit(1)
        )
        items.append(
            PatientListItem(
                id=p.id,
                medical_record_no=p.medical_record_no,
                full_name=p.full_name,
                sex=p.sex,
                latest_bmd=latest_bmd,
                study_count=count or 0,
            )
        )
    return items


@router.post("", response_model=PatientOut, status_code=201)
async def create_patient(
    payload: PatientCreate,
    doctor: Doctor = Depends(get_current_doctor),
    db: AsyncSession = Depends(get_db),
):
    # 환자번호(MRN) 중복 사전 확인 → 친절한 에러 메시지
    existing = await db.scalar(
        select(Patient).where(
            Patient.medical_record_no == payload.medical_record_no
        )
    )
    if existing:
        raise HTTPException(
            status_code=400,
            detail=f"Patient number '{payload.medical_record_no}' is already registered.",
        )
    patient = Patient(**payload.model_dump(), doctor_id=doctor.id)
    db.add(patient)
    await db.commit()
    await db.refresh(patient)
    return patient


@router.get("/{patient_id}", response_model=PatientOut)
async def get_patient(
    patient_id: str,
    doctor: Doctor = Depends(get_current_doctor),
    db: AsyncSession = Depends(get_db),
):
    patient = await db.get(Patient, patient_id)
    if not patient or patient.doctor_id != doctor.id:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient


@router.patch("/{patient_id}", response_model=PatientOut)
async def update_patient(
    patient_id: str,
    payload: PatientUpdate,
    doctor: Doctor = Depends(get_current_doctor),
    db: AsyncSession = Depends(get_db),
):
    patient = await db.get(Patient, patient_id)
    if not patient or patient.doctor_id != doctor.id:
        raise HTTPException(status_code=404, detail="Patient not found")
    for k, v in payload.model_dump(exclude_unset=True).items():
        setattr(patient, k, v)
    await db.commit()
    await db.refresh(patient)
    return patient


@router.delete("/{patient_id}", status_code=204)
async def delete_patient(
    patient_id: str,
    doctor: Doctor = Depends(get_current_doctor),
    db: AsyncSession = Depends(get_db),
):
    """환자 삭제. 환자의 모든 스터디/측정 및 관련 파일도 함께 정리."""
    patient = await db.get(Patient, patient_id)
    if not patient or patient.doctor_id != doctor.id:
        raise HTTPException(status_code=404, detail="Patient not found")

    import shutil

    # 이 환자의 모든 스터디 파일 정리
    studies = (
        await db.scalars(
            select(XrayStudy).where(XrayStudy.patient_id == patient_id)
        )
    ).all()
    for s in studies:
        try:
            dcm = settings.dicom_dir / s.dicom_path
            if dcm.exists():
                dcm.unlink()
        except OSError:
            pass
        try:
            derived = settings.derived_dir / s.id
            if derived.exists():
                shutil.rmtree(derived, ignore_errors=True)
        except OSError:
            pass
    # 환자 디렉토리(있으면) 정리
    try:
        pdir = settings.dicom_dir / patient_id
        if pdir.exists():
            shutil.rmtree(pdir, ignore_errors=True)
    except OSError:
        pass

    await db.delete(patient)  # studies/measurements 등 cascade 삭제
    await db.commit()
    return None
