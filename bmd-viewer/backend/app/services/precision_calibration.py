"""정밀도(LSC) 보정 핵심 로직 — 수동 버튼(POST /precision/calibrate)과
스터디 완료 시 자동 트리거(inference_worker) 양쪽이 공유한다.

과거엔 "최근 8개 스터디"만 표본으로 썼는데, 재위치 섭동 재측정이 몇 건만
실패(스킵)해도 표본이 n=3까지 떨어져 LSC%가 34.7% -> 29.6% -> 7.4%로 크게
흔들리는 게 실측으로 확인됐다. 완료된 스터디 "전체"를 표본으로 쓰도록 바꾸되
(스터디가 아주 많아졌을 때 재보정 한 번이 무한정 오래 걸리지 않도록 상한만
넉넉하게 둔다), 스터디가 완료될 때마다 자동으로 다시 계산해 버튼을 직접
누르지 않아도 최신 상태를 유지한다.
"""
from __future__ import annotations

import asyncio
import logging

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.session import AsyncSessionLocal
from app.models import PrecisionCalibration, StudyStatus, XrayStudy
from app.services.inference.engine import get_engine, run_blocking

logger = logging.getLogger(__name__)

# 표본을 "최근 N개"로 임의로 제한하지 않고 완료된 스터디 전체를 쓰되, 스터디
# 수가 아주 많아졌을 때 재보정 한 번이 무한정 오래 걸리지 않도록(스터디당
# n_repeats회 재측정) 상한만 안전장치로 둔다.
CALIBRATION_SAMPLE_CAP = 60

# 마지막 보정 이후 완료된 스터디가 이 값 이상 쌓였을 때만 자동으로 재보정한다
# -- 표본이 커진 만큼 재보정 자체 비용도 커져서, 스터디 1건 완료마다 매번
# 돌리면 서버 부하가 과도해진다.
AUTO_CALIBRATE_MIN_NEW_STUDIES = 3

# 여러 파일을 한꺼번에 업로드하면 각 스터디의 process_study가 거의 동시에
# 끝난다. 완료될 때마다 바로 재보정을 돌리면 업로드가 아직 끝나지 않았는데도
# 여러 번 겹쳐 돌게 되므로, 마지막 완료로부터 이만큼 조용해질 때까지 기다렸다가
# 딱 한 번만 실행한다 -- 새 완료가 들어오면 타이머가 다시 미뤄진다(디바운스).
AUTO_CALIBRATE_DEBOUNCE_SECONDS = 20.0

_pending_task: asyncio.Task | None = None


async def get_latest_calibration(db: AsyncSession) -> PrecisionCalibration | None:
    return (
        await db.execute(
            select(PrecisionCalibration)
            .order_by(PrecisionCalibration.computed_at.desc())
            .limit(1)
        )
    ).scalar_one_or_none()


async def run_calibration(db: AsyncSession) -> PrecisionCalibration | None:
    """완료된 스터디(최대 CALIBRATION_SAMPLE_CAP개)를 재위치 섭동으로 재측정해
    LSC%를 재산출하고 저장한다. 엔진이 stub이거나 표본이 2개 미만이면 None.
    """
    engine = get_engine()
    if not hasattr(engine, "measure_precision"):
        return None

    studies = (
        (
            await db.execute(
                select(XrayStudy)
                .where(XrayStudy.status == StudyStatus.COMPLETED)
                .order_by(XrayStudy.uploaded_at.desc())
                .limit(CALIBRATION_SAMPLE_CAP)
            )
        )
        .scalars()
        .all()
    )
    if len(studies) < 2:
        return None

    dicom_paths = [settings.dicom_dir / s.dicom_path for s in studies]
    result = await run_blocking(engine.measure_precision, dicom_paths)
    if result is None:
        return None

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


async def _new_completed_studies_since(
    db: AsyncSession, latest: PrecisionCalibration | None
) -> int:
    stmt = select(func.count()).select_from(XrayStudy).where(
        XrayStudy.status == StudyStatus.COMPLETED
    )
    if latest is not None:
        stmt = stmt.where(XrayStudy.uploaded_at > latest.computed_at)
    return (await db.execute(stmt)).scalar_one()


async def _debounced_run() -> None:
    try:
        await asyncio.sleep(AUTO_CALIBRATE_DEBOUNCE_SECONDS)
    except asyncio.CancelledError:
        return  # 대기 중 다른 스터디가 완료돼 타이머가 다시 미뤄짐 -- 정상 취소

    try:
        async with AsyncSessionLocal() as db:
            latest = await get_latest_calibration(db)
            threshold = 2 if latest is None else AUTO_CALIBRATE_MIN_NEW_STUDIES
            new_count = await _new_completed_studies_since(db, latest)
            if new_count < threshold:
                return
            await run_calibration(db)
    except Exception:  # noqa: BLE001 — 자동 보정 실패가 업로드/추론을 건드리면 안 됨
        logger.exception("자동 정밀도 보정 실패 (다음 스터디 완료 시 다시 시도됨)")


def schedule_auto_calibration() -> None:
    """스터디 완료 직후 호출한다. 여러 파일이 한꺼번에 끝나도, 마지막 완료로부터
    AUTO_CALIBRATE_DEBOUNCE_SECONDS만큼 조용해진 뒤 딱 한 번만 실제로 돈다."""
    global _pending_task
    if _pending_task is not None and not _pending_task.done():
        _pending_task.cancel()
    _pending_task = asyncio.create_task(_debounced_run())
