"""X-ray 스터디 엔드포인트 - 업로드(비동기 추론), 이력, 상세."""
from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    UploadFile,
)
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.api.deps import get_current_doctor
from app.core.config import settings
from app.db.session import get_db
from app.models import (
    BmdMeasurement,
    Doctor,
    Patient,
    StudyStatus,
    XrayStudy,
)
from app.schemas import XrayStudyDetail, XrayStudyListItem
from app.workers.inference_worker import process_study

router = APIRouter()


async def _assert_owns_patient(
    db: AsyncSession, doctor: Doctor, patient_id: str
) -> Patient:
    patient = await db.get(Patient, patient_id)
    if not patient or patient.doctor_id != doctor.id:
        raise HTTPException(status_code=404, detail="환자를 찾을 수 없습니다")
    return patient


@router.get("/patients/{patient_id}/studies",
            response_model=list[XrayStudyListItem])
async def list_studies(
    patient_id: str,
    doctor: Doctor = Depends(get_current_doctor),
    db: AsyncSession = Depends(get_db),
):
    """환자의 X-ray 이력 목록 (최신순)."""
    await _assert_owns_patient(db, doctor, patient_id)
    studies = (
        await db.scalars(
            select(XrayStudy)
            .where(XrayStudy.patient_id == patient_id)
            .options(selectinload(XrayStudy.measurement))
            .order_by(XrayStudy.uploaded_at.desc())
        )
    ).all()
    return [
        XrayStudyListItem(
            id=s.id,
            acquired_at=s.acquired_at,
            uploaded_at=s.uploaded_at,
            status=s.status,
            modality=s.modality,
            preview_path=s.preview_path,
            bmd_value=s.measurement.bmd_value if s.measurement else None,
        )
        for s in studies
    ]


@router.post("/patients/{patient_id}/studies",
             response_model=XrayStudyDetail, status_code=202)
async def upload_study(
    patient_id: str,
    background: BackgroundTasks,
    file: UploadFile = File(...),
    doctor: Doctor = Depends(get_current_doctor),
    db: AsyncSession = Depends(get_db),
):
    """DICOM 업로드 → 저장 → 비동기 추론 트리거 (202 Accepted)."""
    await _assert_owns_patient(db, doctor, patient_id)

    study_id = str(uuid.uuid4())
    rel_path = f"{patient_id}/{study_id}.dcm"
    abs_path = settings.dicom_dir / rel_path
    abs_path.parent.mkdir(parents=True, exist_ok=True)

    content = await file.read()
    abs_path.write_bytes(content)

    study = XrayStudy(
        id=study_id,
        patient_id=patient_id,
        dicom_path=rel_path,
        status=StudyStatus.UPLOADED,
    )
    db.add(study)
    await db.commit()

    # 비동기 추론 트리거 (worker_backend=celery면 큐로 교체)
    background.add_task(process_study, study_id)

    # 업로드 직후에는 measurement가 없으므로 명시적으로 구성 (lazy-load 회피)
    return XrayStudyDetail(
        id=study.id,
        patient_id=study.patient_id,
        dicom_path=study.dicom_path,
        preview_path=study.preview_path,
        acquired_at=study.acquired_at,
        modality=study.modality,
        status=study.status,
        error_message=study.error_message,
        uploaded_at=study.uploaded_at,
        measurement=None,
    )


@router.get("/studies/{study_id}", response_model=XrayStudyDetail)
async def get_study(
    study_id: str,
    doctor: Doctor = Depends(get_current_doctor),
    db: AsyncSession = Depends(get_db),
):
    """스터디 상세: 원본 + 추출 척추 + L4 BMD + XAI."""
    study = await db.scalar(
        select(XrayStudy)
        .where(XrayStudy.id == study_id)
        .options(
            selectinload(XrayStudy.measurement).selectinload(
                BmdMeasurement.segments
            ),
            selectinload(XrayStudy.measurement).selectinload(
                BmdMeasurement.xai_factors
            ),
        )
    )
    if not study:
        raise HTTPException(status_code=404, detail="스터디를 찾을 수 없습니다")
    await _assert_owns_patient(db, doctor, study.patient_id)
    return study
