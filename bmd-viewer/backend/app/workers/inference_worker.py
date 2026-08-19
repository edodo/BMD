"""비동기 추론 워커.

업로드 직후 FastAPI BackgroundTasks로 호출된다.
worker_backend 설정을 "celery"로 바꾸면 동일 인터페이스로 큐 기반 전환 가능.
"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.config import settings
from app.db.session import AsyncSessionLocal
from app.models import (
    BmdMeasurement,
    StudyStatus,
    VertebraSegment,
    XaiFactor,
    XrayStudy,
)
from app.services.inference.engine import (
    InferenceResult,
    PartialInferenceError,
    get_engine,
)


async def _delete_existing_measurement(db: AsyncSession, study_id: str) -> None:
    """재계산 전, 이 스터디의 기존 measurement(및 cascade로 segments/xai_factors)를
    명시적으로 로드해 삭제한다. `study.measurement = ...` 재할당 방식은 async 세션에서
    이전 값을 지연로드해야 해 MissingGreenlet 위험이 있어 피한다 — 대신 FK로 직접
    조회 후 삭제하고, 새 measurement는 study_id를 채워 별도로 add한다."""
    old = await db.scalar(
        select(BmdMeasurement)
        .where(BmdMeasurement.study_id == study_id)
        .options(
            selectinload(BmdMeasurement.segments),
            selectinload(BmdMeasurement.xai_factors),
        )
    )
    if old is not None:
        await db.delete(old)
        await db.flush()


def _apply_result(study: XrayStudy, result: InferenceResult) -> BmdMeasurement:
    """InferenceResult를 study/measurement ORM 필드로 반영한다.

    최초 업로드(process_study)와 뷰 수동 재계산(recompute_view)이 공유하는
    단일 책임 함수 — 두 경로가 결과를 DB에 쓰는 방식이 어긋나지 않게 한다.
    호출자가 (필요하다면) 기존 measurement를 먼저 지우고, 반환된 measurement를
    db.add()해야 한다.
    """
    study.preview_path = result.preview_path
    study.modality = result.modality
    study.view_position = result.view_position
    study.view_source = result.view_source
    if result.acquired_at:
        from datetime import datetime

        try:
            study.acquired_at = datetime.fromisoformat(result.acquired_at)
        except ValueError:
            pass
    if result.dicom_meta:
        import json

        study.dicom_meta = json.dumps(result.dicom_meta, ensure_ascii=False)

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
        xai_bar_l1l5_path=result.xai_bar_l1l5_path,
        xai_air_soft_path=result.xai_air_soft_path,
        density_low=result.density_low,
        density_high=result.density_high,
        roi_mean_intensity=result.roi_mean_attenuation,
        reliable=result.reliable,
        reliability_warning=result.reliability_warning,
        l4_density_grid=result.l4_density_grid,
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
            bar=s.bar,
            qc_status=s.qc_status,
            qc_message=s.qc_message,
            glcm_contrast=s.glcm_contrast,
            glcm_correlation=s.glcm_correlation,
            glcm_energy=s.glcm_energy,
            glcm_homogeneity=s.glcm_homogeneity,
            glcm_entropy=s.glcm_entropy,
            vario_slope=s.vario_slope,
            fractal_dim=s.fractal_dim,
            anomaly_mse=s.anomaly_mse,
            anomaly_pct=s.anomaly_pct,
            anomaly_shap=s.anomaly_shap,
            bar_z=s.bar_z,
            bhi_z=s.bhi_z,
            bhi_pct=s.bhi_pct,
            category=s.category,
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
    return measurement


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
            measurement = _apply_result(study, result)
            db.add(measurement)
            study.status = StudyStatus.COMPLETED
            await db.commit()

        except PartialInferenceError as exc:
            # L4 측정은 실패했지만 원본/추출 오버레이는 살려서 화면에 보여준다.
            study.status = StudyStatus.FAILED
            study.error_message = str(exc)
            if exc.preview_path:
                study.preview_path = exc.preview_path
            if exc.overlay_path:
                study.overlay_path = exc.overlay_path
            await db.commit()

        except Exception as exc:  # noqa: BLE001
            study.status = StudyStatus.FAILED
            study.error_message = str(exc)
            await db.commit()


async def recompute_view(db: AsyncSession, study: XrayStudy, view: str) -> None:
    """의사가 화면에서 AP/LA를 수동 지정했을 때, 그 뷰로 측정을 다시 실행한다.

    자동판별(ViewPosition 태그/휴리스틱)이 실제 촬영 방향과 다를 수 있으므로
    (v18_spec.md Sec.4.5), 의사의 육안 판단으로 덮어써 재계산할 수 있게 한다.
    기존 measurement/segments/xai_factors는 새 결과로 교체된다(이력이 아니라
    '현재 뷰 판정에서의 측정값'이므로 최신 1건만 유지 — process_study와 동일
    스키마).

    호출자(엔드포인트)의 요청-스코프 세션을 그대로 받는다 — process_study처럼
    새 세션을 열면, 엔드포인트가 이미 들고 있는 세션의 identity map이 갱신되지
    않아(expire_on_commit=False) 응답이 재계산 이전의 stale한 study를 돌려줄
    위험이 있다. 실패 시 조용히 넘어가지 않고 예외를 올려 엔드포인트가 에러로
    응답하게 한다.
    """
    engine = get_engine()
    out_dir = settings.derived_dir / study.id
    result = engine.run(
        dicom_path=settings.dicom_dir / study.dicom_path,
        output_dir=out_dir,
        view_override=view,
    )
    await _delete_existing_measurement(db, study.id)
    measurement = _apply_result(study, result)
    db.add(measurement)
    await db.commit()
