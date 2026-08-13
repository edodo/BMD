"""기존 스터디를 v17 BAR 산식으로 재추론한다.

왜 필요한가
-----------
v17에서 주 지표가 0..1 proxy 점수 -> BAR(배경 대비 상대 감쇠)로 바뀌었다. 두 산식은
스케일이 호환되지 않으므로, 재추론하지 않으면 한 환자의 추세선에 서로 다른 단위가
섞여 임상적으로 오해를 부른다.

사용법 (backend 디렉터리에서)
-----------------------------
    python -m scripts.reinfer_bar              # DRY-RUN: 무엇이 바뀌는지만 출력
    python -m scripts.reinfer_bar --apply      # 실제 DB 갱신
    python -m scripts.reinfer_bar --apply --only-legacy   # 구버전 레코드만

안전장치
--------
* 기본이 dry-run이다. --apply 없이는 DB를 건드리지 않는다.
* 원본 DICOM이 없으면 그 스터디는 건너뛰고 사유를 출력한다.
* 실패한 스터디가 있어도 나머지는 계속 처리하고, 마지막에 요약을 낸다.
* 갱신 전 bmd_value/model_version을 로그로 남겨 되돌릴 근거를 남긴다.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.core.config import settings
from app.db.session import AsyncSessionLocal
from app.models import BmdMeasurement, Study
from app.services.inference.engine import get_engine

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("reinfer")

BAR_VERSION_MARK = "-bar"      # v17 이후 MODEL_VERSION 에 포함되는 표식


def _dicom_path(study: Study) -> Path | None:
    """스터디의 원본 DICOM 경로. 스키마 차이를 흡수하기 위해 후보를 순서대로 시도."""
    for attr in ("dicom_path", "file_path", "storage_path", "path"):
        v = getattr(study, attr, None)
        if v:
            p = Path(v)
            if not p.is_absolute():
                p = settings.dicom_dir / p
            if p.exists():
                return p
    # 저장 규칙 폴백: studies.py 업로드가 쓰는 "<patient_id>/<study_id>.dcm".
    # (업로드는 원본이 .dicom이어도 .dcm으로 정규화해 저장한다)
    pid = getattr(study, "patient_id", None)
    for cand in (
        settings.dicom_dir / f"{pid}/{study.id}.dcm" if pid else None,
        settings.dicom_dir / f"{study.id}.dcm",
    ):
        if cand is not None and cand.exists():
            return cand
    return None


async def main(apply: bool, only_legacy: bool) -> None:
    engine = get_engine()
    log.info("engine=%s model_version=%s",
             type(engine).__name__, getattr(engine, "MODEL_VERSION", "?"))
    if apply and BAR_VERSION_MARK not in str(getattr(engine, "MODEL_VERSION", "")):
        log.error("현재 엔진이 BAR 버전이 아니다. BMD_INFERENCE_ENGINE=yolo 인지 확인하라.")
        return

    ok = skipped = failed = 0
    async with AsyncSessionLocal() as session:
        rows = (
            await session.execute(
                select(Study).options(selectinload(Study.measurement))
            )
        ).scalars().all()
        log.info("대상 스터디 %d건 (apply=%s, only_legacy=%s)", len(rows), apply, only_legacy)

        for st in rows:
            m: BmdMeasurement | None = getattr(st, "measurement", None)
            if m is None:
                skipped += 1
                continue
            if only_legacy and BAR_VERSION_MARK in (m.model_version or ""):
                skipped += 1
                continue

            src = _dicom_path(st)
            if src is None:
                log.warning("[skip] %s: 원본 DICOM 없음", st.id)
                skipped += 1
                continue

            out_dir = settings.derived_dir / str(st.id)
            try:
                res = engine.run(src, out_dir)
            except Exception as exc:  # noqa: BLE001 - 한 건 실패가 전체를 막지 않게
                log.warning("[fail] %s: %s: %s", st.id, type(exc).__name__, exc)
                failed += 1
                continue

            log.info("[ok] %s: %.3f (%s) -> %.3f (%s)%s",
                     st.id, m.bmd_value, m.model_version or "?",
                     res.bmd_value, res.model_version,
                     "" if res.reliable else "  [UNRELIABLE]")

            if apply:
                m.bmd_value = res.bmd_value
                m.model_version = res.model_version
                m.confidence = res.confidence
                m.density_low = res.density_low
                m.density_high = res.density_high
                m.roi_mean_intensity = res.roi_mean_attenuation
                m.l4_density_grid = res.l4_density_grid
                m.reliable = res.reliable
                m.reliability_warning = res.reliability_warning
                if res.preview_path:
                    m.preview_path = res.preview_path
                if res.gradcam_path:
                    m.gradcam_path = res.gradcam_path
                if res.l4_crop_path:
                    m.l4_crop_path = res.l4_crop_path
                if res.xai_bar_l1l5_path:
                    m.xai_bar_l1l5_path = res.xai_bar_l1l5_path
            ok += 1

        if apply:
            await session.commit()
            log.info("커밋 완료.")
        else:
            log.info("DRY-RUN — DB는 변경되지 않았다. 실제 반영하려면 --apply 를 붙여라.")

    log.info("요약: 갱신 %d / 건너뜀 %d / 실패 %d", ok, skipped, failed)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="기존 스터디를 v17 BAR 산식으로 재추론")
    ap.add_argument("--apply", action="store_true", help="실제로 DB를 갱신한다 (기본은 dry-run)")
    ap.add_argument("--only-legacy", action="store_true",
                    help="아직 BAR로 재추론되지 않은 레코드만 처리")
    a = ap.parse_args()
    asyncio.run(main(a.apply, a.only_legacy))
