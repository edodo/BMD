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
from sqlalchemy import func, select
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
from app.schemas import (
    StudyDateUpdate,
    StudyNoteUpdate,
    ViewOverrideIn,
    XrayStudyDetail,
    XrayStudyListItem,
)
from app.services.inference.engine import PartialInferenceError
from app.workers.inference_worker import process_study, recompute_view

router = APIRouter()


async def _assert_owns_patient(
    db: AsyncSession, doctor: Doctor, patient_id: str
) -> Patient:
    patient = await db.get(Patient, patient_id)
    if not patient or patient.doctor_id != doctor.id:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient


@router.get("/patients/{patient_id}/studies",
            response_model=list[XrayStudyListItem])
async def list_studies(
    patient_id: str,
    doctor: Doctor = Depends(get_current_doctor),
    db: AsyncSession = Depends(get_db),
):
    """환자의 X-ray 이력 목록 (촬영일순 — 지정 안 됐으면 업로드일로 대체).

    acquired_at은 기본적으로 DICOM 메타에서만 채워지지만(추론 완료 후),
    의사가 PATCH /studies/{id}/date로 직접 지정/수정할 수도 있다. 어느
    쪽으로도 값이 없으면(추론 전/실패) uploaded_at으로 정렬 — 지금까지의
    동작(업로드순)과 동일하게 유지.
    """
    await _assert_owns_patient(db, doctor, patient_id)
    order_key = func.coalesce(XrayStudy.acquired_at, XrayStudy.uploaded_at)
    studies = (
        await db.scalars(
            select(XrayStudy)
            .where(XrayStudy.patient_id == patient_id)
            .options(selectinload(XrayStudy.measurement))
            .order_by(order_key.desc())
        )
    ).all()
    return [
        XrayStudyListItem(
            id=s.id,
            original_filename=s.original_filename,
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
        original_filename=file.filename,
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
        original_filename=study.original_filename,
        preview_path=study.preview_path,
        acquired_at=study.acquired_at,
        modality=study.modality,
        dicom_meta=study.dicom_meta,
        note=study.note,
        note_updated_at=study.note_updated_at,
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
        raise HTTPException(status_code=404, detail="Study not found")
    await _assert_owns_patient(db, doctor, study.patient_id)
    return study

@router.patch("/studies/{study_id}/note", response_model=XrayStudyDetail)
async def update_study_note(
    study_id: str,
    payload: StudyNoteUpdate,
    doctor: Doctor = Depends(get_current_doctor),
    db: AsyncSession = Depends(get_db),
):
    """스터디(이력)별 의사 메모 저장/수정."""
    study = await db.get(XrayStudy, study_id)
    if not study:
        raise HTTPException(status_code=404, detail="Study not found")
    await _assert_owns_patient(db, doctor, study.patient_id)
    from datetime import datetime, timezone

    study.note = payload.note
    study.note_updated_at = datetime.now(timezone.utc)
    await db.commit()
    # 상세 재조회 (관계 포함)
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
    return study


@router.patch("/studies/{study_id}/date", response_model=XrayStudyDetail)
async def update_study_date(
    study_id: str,
    payload: StudyDateUpdate,
    doctor: Doctor = Depends(get_current_doctor),
    db: AsyncSession = Depends(get_db),
):
    """촬영일(acquired_at) 수동 지정/수정 — History/비교 정렬이 이 값을 따른다."""
    study = await db.get(XrayStudy, study_id)
    if not study:
        raise HTTPException(status_code=404, detail="Study not found")
    await _assert_owns_patient(db, doctor, study.patient_id)
    study.acquired_at = payload.acquired_at
    await db.commit()
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
    return study


@router.post("/studies/{study_id}/view", response_model=XrayStudyDetail)
async def override_view(
    study_id: str,
    payload: ViewOverrideIn,
    doctor: Doctor = Depends(get_current_doctor),
    db: AsyncSession = Depends(get_db),
):
    """의사가 AP/LA를 수동 지정 -> 그 뷰로 즉시 재계산 (v18_spec.md Sec.4.5).

    자동판별(DICOM ViewPosition 태그, 없으면 AP 기본)이 실제 촬영 방향과
    다를 때 화면에서 바로잡는다. 동기 요청 — 재추론 1건이라 몇 초 내 끝난다.
    """
    study = await db.get(XrayStudy, study_id)
    if not study:
        raise HTTPException(status_code=404, detail="Study not found")
    await _assert_owns_patient(db, doctor, study.patient_id)
    if study.status != StudyStatus.COMPLETED:
        raise HTTPException(
            status_code=409,
            detail=f"Study is not completed yet (status={study.status.value}).",
        )

    try:
        await recompute_view(db, study, payload.view)
    except PartialInferenceError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

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
    return study


@router.delete("/studies/{study_id}", status_code=204)
async def delete_study(
    study_id: str,
    doctor: Doctor = Depends(get_current_doctor),
    db: AsyncSession = Depends(get_db),
):
    """X-ray 스터디(이력) 삭제. 원본 DICOM과 파생 산출물 파일도 정리."""
    study = await db.get(XrayStudy, study_id)
    if not study:
        raise HTTPException(status_code=404, detail="Study not found")
    await _assert_owns_patient(db, doctor, study.patient_id)

    # 파일 정리 (원본 DICOM + 파생 디렉토리)
    import shutil

    try:
        dcm = settings.dicom_dir / study.dicom_path
        if dcm.exists():
            dcm.unlink()
    except OSError:
        pass
    try:
        derived = settings.derived_dir / study.id
        if derived.exists():
            shutil.rmtree(derived, ignore_errors=True)
    except OSError:
        pass

    await db.delete(study)  # measurement/segments/xai는 cascade로 삭제
    await db.commit()
    return None
