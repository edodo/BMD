"""같은 L4 ROI에 대해 air 기준 BAR과 disc(추간판) 기준 BAR을 나란히 비교하는 도구.

air 기준(현재 프로덕션 공식, `_air_reference`)이 "프레임에 air가 찍혔는지"에
따라 흔들린다는 문제 제기에서, 대안으로 L4 바로 위/아래 추간판(disc) 공간을
soft-tissue 다음 두 번째 기준점으로 쓰는 안을 논의했다. 이 스크립트는 그 안을
실제로 구현해, 같은 파일에서 두 공식을 동시에 계산해 한 줄씩 비교한다.

공식
----
  분자(A_trab - A_soft)는 두 버전이 완전히 동일하다 — 다른 건 분모의 두 번째
  기준점뿐이다.
    BAR_air  = (A_trab - A_soft) / (A_soft - A_air)   <- 지금 프로덕션 공식
    BAR_disc = (A_trab - A_soft) / (A_soft - A_disc)  <- 이 스크립트가 새로 추가

  A_disc는 L4-L3(위쪽) / L4-L5(아래쪽) 사이 간격을, 두 척추 bbox의 교집합
  가로폭 중앙 60%로 좁혀(후관절/횡돌기 오염 회피, `soft_tissue_ref`의
  '오염은 감쇠를 더하기만 한다'는 논리와 동일) 뽑은 뒤 하위 25%
  분위수(SOFT_TISSUE_PCT, soft_tissue_ref와 동일 통계량)를 쓴다. 표본이
  DISC_MIN_PX(50px, soft_tissue_ref의 최소 표본 기준과 동일) 미만이면(디스크
  협착 등으로 그 방향은 못 쓰는 경우) 그 방향은 버리고, 위/아래 둘 다 없으면
  disc 기준값 자체를 "no usable disc region"으로 표시하고 BAR_disc는 계산하지
  않는다 — 조용히 이상한 숫자를 내지 않기 위함(이전에 논의한 QC 가드).

사용법 (backend 디렉터리, 모델/torch가 있는 환경)
------------------------------------------------
    python scripts/debug_bar_air_vs_disc.py <dicom 파일 또는 폴더...>
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

_BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from app.services.inference.yolo_engine import (
    CLASS_NAMES,
    CONF,
    IMGSZ,
    YoloBmdEngine,
)

L3_CLASS_ID = CLASS_NAMES.index("L3")
L5_CLASS_ID = CLASS_NAMES.index("L5")
DISC_MIN_PX = 50        # soft_tissue_ref()의 최소 표본 기준(50px)과 동일
DISC_TRIM_FRAC = 0.2    # 두 척추 bbox 교집합 폭의 좌우 20%씩 잘라 중앙만 사용

DICOM_EXTS = {".dcm", ".dicom"}


def collect_files(args: list[str]) -> list[Path]:
    files: list[Path] = []
    for a in args:
        p = Path(a)
        if p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file() and f.suffix.lower() in DICOM_EXTS:
                    files.append(f)
        elif p.is_file():
            files.append(p)
        else:
            print(f"[warn] not found: {a}")
    return files


def _gap_band_mask(H, W, box_upper, box_lower, bone_dil, trim=DISC_TRIM_FRAC):
    """box_upper(위쪽 척추)와 box_lower(아래쪽 척추) 사이 간격의 중앙 밴드.

    y축: box_upper 하단 ~ box_lower 상단. x축: 두 bbox 교집합 폭의 중앙
    (1 - 2*trim)만 사용해 후관절/횡돌기 쪽 오염을 피한다. bone_dil(전체
    척추 팽창 마스크)과 겹치는 픽셀은 제외 — 세그멘테이션이 살짝 넓게
    잡혔어도 뼈가 섞여 들어가지 않게."""
    ux0, uy0, ux1, uy1 = box_upper
    lx0, ly0, lx1, ly1 = box_lower
    y0, y1 = int(round(uy1)), int(round(ly0))
    if y1 <= y0:
        return None
    x0, x1 = max(ux0, lx0), min(ux1, lx1)
    if x1 <= x0:
        return None
    w = x1 - x0
    x0, x1 = x0 + trim * w, x1 - trim * w
    x0, x1 = int(round(x0)), int(round(x1))
    if x1 <= x0:
        return None
    m = np.zeros((H, W), dtype=bool)
    m[max(0, y0) : min(H, y1), max(0, x0) : min(W, x1)] = True
    m &= bone_dil == 0
    return m


def disc_reference(gray_lin, boxes, classes, confs, l4_i, bone_dil, H, W):
    """L4 위/아래 disc space의 하위 분위수 밝기(gray, [0,1]).

    Returns (disc_gray | None, detail_str)."""

    def _best(cls_id: int) -> int | None:
        idx = [i for i in range(len(classes)) if classes[i] == cls_id]
        return max(idx, key=lambda i: confs[i]) if idx else None

    l3_i, l5_i = _best(L3_CLASS_ID), _best(L5_CLASS_ID)
    l4_box = boxes[l4_i]
    candidates: list[float] = []
    parts: list[str] = []

    if l3_i is not None:
        m = _gap_band_mask(H, W, boxes[l3_i], l4_box, bone_dil)
        if m is not None and m.sum() >= DISC_MIN_PX:
            candidates.append(
                float(np.percentile(gray_lin[m], YoloBmdEngine.SOFT_TISSUE_PCT))
            )
            parts.append(f"upper(n={int(m.sum())})")
        else:
            parts.append("upper(too small)" if m is not None else "upper(no overlap)")
    else:
        parts.append("upper(L3 not detected)")

    if l5_i is not None:
        m = _gap_band_mask(H, W, l4_box, boxes[l5_i], bone_dil)
        if m is not None and m.sum() >= DISC_MIN_PX:
            candidates.append(
                float(np.percentile(gray_lin[m], YoloBmdEngine.SOFT_TISSUE_PCT))
            )
            parts.append(f"lower(n={int(m.sum())})")
        else:
            parts.append("lower(too small)" if m is not None else "lower(no overlap)")
    else:
        parts.append("lower(L5 not detected)")

    detail = "+".join(parts)
    if not candidates:
        return None, detail
    return float(np.mean(candidates)), detail


def measure_both(engine: YoloBmdEngine, path: Path) -> dict | None:
    model = engine._ensure_model()
    rgb, dens, H, W, *_ = YoloBmdEngine._load_dicom(path)
    response = YoloBmdEngine._detector_response(path)

    r = model.predict(rgb, conf=CONF, imgsz=IMGSZ, verbose=False)[0]
    if r.masks is None or len(r.boxes) == 0:
        return None
    boxes = r.boxes.xyxy.cpu().numpy()
    classes = r.boxes.cls.cpu().numpy().astype(int)
    confs = r.boxes.conf.cpu().numpy()
    l4_i, _status = YoloBmdEngine._pick_l4_index(boxes, classes, confs)
    if l4_i is None:
        return None

    tag_view = YoloBmdEngine._view_position(path)
    view = (
        tag_view
        if tag_view != "unknown"
        else YoloBmdEngine._view_from_shape(boxes, r.masks.xy, H, W)
    )

    valid = YoloBmdEngine._valid_tissue_mask(dens)
    bone_union = np.zeros((H, W), np.uint8)
    for i in range(len(boxes)):
        bone_union |= YoloBmdEngine._poly_to_mask(r.masks.xy[i], H, W, shrink=1.0)
    bone_dil = cv2.dilate(bone_union, np.ones((7, 7), np.uint8), 1)
    gray_lin = YoloBmdEngine._gray_lin(dens)
    crop_box = YoloBmdEngine._crop_box(H, W, boxes[l4_i], pad=0.6)

    bar_air, qc_status, _msg, _roi, bar_parts = YoloBmdEngine._measure_vertebra_bar(
        gray_lin, dens, r.masks.xy[l4_i], H, W, valid, bone_dil, response, crop_box, view
    )

    out: dict = {"view": view, "qc_status": qc_status, "bar_air": bar_air}
    if bar_air is None or "a_soft" not in bar_parts:
        out["a_soft"] = bar_parts.get("a_soft")
        out["a_air"] = bar_parts.get("a_air")
        out["disc_detail"] = "skipped (air/soft unavailable)"
        out["bar_disc"] = None
        return out

    a_soft = bar_parts["a_soft"]
    a_air = bar_parts["a_air"]
    margin_air = bar_parts["margin"]
    a_trab = bar_air * margin_air + a_soft  # bar_score 내부 계산의 역산
    out.update(a_trab=a_trab, a_soft=a_soft, a_air=a_air)

    disc_gray, disc_detail = disc_reference(
        gray_lin, boxes, classes, confs, l4_i, bone_dil, H, W
    )
    out["disc_detail"] = disc_detail
    if disc_gray is None:
        out["bar_disc"] = None
        return out

    a_disc = float(
        YoloBmdEngine._to_attenuation(np.array([[disc_gray]], np.float32), response)[0, 0]
    )
    margin_disc = a_soft - a_disc
    out["a_disc"] = a_disc
    out["margin_disc"] = margin_disc
    out["bar_disc"] = (
        (a_trab - a_soft) / margin_disc
        if margin_disc > YoloBmdEngine.QC_MIN_SOFT_MARGIN
        else None
    )
    return out


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python scripts/debug_bar_air_vs_disc.py <dicom files or folders...>")
        raise SystemExit(1)

    files = collect_files(sys.argv[1:])
    if not files:
        print("No DICOM files found.")
        raise SystemExit(1)

    engine = YoloBmdEngine()
    rows: list[dict] = []
    for f in files:
        try:
            res = measure_both(engine, f)
        except Exception as exc:  # noqa: BLE001
            print(f"{f.name} : [error] {type(exc).__name__}: {exc}")
            continue

        if res is None:
            print(f"{f.name} : [no L4 detected]")
            continue

        bar_air = res["bar_air"]
        bar_disc = res.get("bar_disc")
        if bar_air is None:
            print(
                f"{f.name} : air BAR unavailable (soft={res.get('a_soft')!r} "
                f"air={res.get('a_air')!r})  disc=[{res['disc_detail']}]"
            )
            continue

        air_str = f"{bar_air:.3f}"
        disc_str = f"{bar_disc:.3f}" if bar_disc is not None else "n/a"
        delta_str = f"{bar_disc - bar_air:+.3f}" if bar_disc is not None else "n/a"
        print(
            f"{f.name} : trab={res['a_trab']:.4f} soft={res['a_soft']:.4f} | "
            f"air={res['a_air']:.4f}->BAR_air={air_str} | "
            f"disc={res.get('a_disc', float('nan')):.4f}->BAR_disc={disc_str} | "
            f"delta={delta_str}  (view={res['view']}, QC={res['qc_status']}, "
            f"disc=[{res['disc_detail']}])"
        )
        rows.append(res)

    usable = [r for r in rows if r.get("bar_air") is not None and r.get("bar_disc") is not None]
    n_disc_missing = sum(
        1 for r in rows if r.get("bar_air") is not None and r.get("bar_disc") is None
    )
    print()
    print(
        f"n_total={len(files)}  n_air_ok={len(rows)}  "
        f"n_both_ok={len(usable)}  n_disc_unavailable={n_disc_missing}"
    )
    if usable:
        air_arr = np.array([r["bar_air"] for r in usable])
        disc_arr = np.array([r["bar_disc"] for r in usable])
        delta = disc_arr - air_arr
        print(
            f"BAR_air : median={np.median(air_arr):.3f}  "
            f"p5={np.percentile(air_arr,5):.3f}  p95={np.percentile(air_arr,95):.3f}"
        )
        print(
            f"BAR_disc: median={np.median(disc_arr):.3f}  "
            f"p5={np.percentile(disc_arr,5):.3f}  p95={np.percentile(disc_arr,95):.3f}"
        )
        print(
            f"delta(disc-air): median={np.median(delta):+.3f}  "
            f"std={delta.std():.3f}  |median| 클수록 두 기준이 다른 값을 낸다는 뜻, "
            "std 클수록 disc가 air보다 불안정하다는 뜻."
        )


if __name__ == "__main__":
    main()
