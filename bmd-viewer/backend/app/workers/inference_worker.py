"""비동기 추론 워커.

업로드 직후 FastAPI BackgroundTasks로 호출된다.
worker_backend 설정을 "celery"로 바꾸면 동일 인터페이스로 큐 기반 전환 가능.
"""
from __future__ import annotations

from pathlib import Path

from sqlalchemy import select

from app.core.config import settings
from app.db.session import AsyncSessionLocal
from app.models import (
    BmdMeasurement,
    StudyStatus,
    VertebraSegment,
    XaiFactor,
    XrayStudy,
)
from app.services.inference.engine import get_engine


async def process_study(study_id: str) -> None:
    """단일 스터디에 대해 추론을 실행하고 결과를 저장한다."""
    async with AsyncSessionLocal() as db:
        study = await db.get(XrayStudy, study_id)
        if study is None:
            return

        study.status = StudyStatus.PROCESSING
        await db.commit()

        try:
            engine = get_engine()
            out_dir = settings.derived_dir / study_id
            result = engine.run(
                dicom_path=settings.dicom_dir / study.dicom_path,
                output_dir=out_dir,
            )

            # 스터디 메타 갱신
            study.preview_path = result.preview_path
            study.modality = result.modality
            if result.acquired_at:
                from datetime import datetime

                try:
                    study.acquired_at = datetime.fromisoformat(
                        result.acquired_at
                    )
                except ValueError:
                    pass
            if result.dicom_meta:
                import json

                study.dicom_meta = json.dumps(
                    result.dicom_meta, ensure_ascii=False
                )

            measurement = BmdMeasurement(
                study_id=study.id,
                target_vertebra=result.target_vertebra,
                bmd_value=result.bmd_value,
                t_score=result.t_score,
                z_score=result.z_score,
                confidence=result.confidence,
                exposure_corrected=result.exposure_corrected,
                gradcam_path=result.gradcam_path,
                l4_crop_path=result.l4_crop_path,
                xai_overlay_path=result.xai_overlay_path,
                xai_l4_cam_path=result.xai_l4_cam_path,
                model_version=result.model_version,
            )
            measurement.segments = [
                VertebraSegment(
                    label=s.label,
                    is_target=s.is_target,
                    bbox_x=s.bbox[0],
                    bbox_y=s.bbox[1],
                    bbox_w=s.bbox[2],
                    bbox_h=s.bbox[3],
                    mask_path=s.mask_path,
                    mean_intensity=s.mean_intensity,
                )
                for s in result.segments
            ]
            measurement.xai_factors = [
                XaiFactor(
                    label=x.label,
                    contribution=x.contribution,
                    description=x.description,
                    rank=i,
                )
                for i, x in enumerate(result.xai_factors)
            ]
            db.add(measurement)
            study.status = StudyStatus.COMPLETED
            await db.commit()

        except Exception as exc:  # noqa: BLE001
            study.status = StudyStatus.FAILED
            study.error_message = str(exc)
            await db.commit()
