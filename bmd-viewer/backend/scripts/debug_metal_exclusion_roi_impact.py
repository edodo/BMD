"""금속을 trabecular ROI에서 '빼고' 계산하는 게 실제로 값을 왜곡시키는지 확인한다.

배경
----
사용자가 과거에 "금속이 있으면 그 공간만큼 제외해서 계산했더니 값이 크게
뒤틀렸다"고 보고했다. B2(`detect_metal_mask`/`vertebra_roi`)를 프로덕션에
이식하기 전에, 이번에 실측한 진짜 핀 데이터로 그 우려가 재현되는지 먼저
확인한다.

측정 (모델 필요)
----------------
검출된 척추마다:
  - roi_before      : 금속 제외 전 trabecular_roi 픽셀 수 (지금 프로덕션의
                       실질 동작과 같음 -- 예전 bmd_metal_ratio는 실측으로
                       이미 no-op임이 확인됨)
  - roi_after       : 금속(B2) 제외 후 픽셀 수
  - frac_removed    : (roi_before - roi_after) / roi_before
  - n_fragments     : 제외 후 ROI의 연결성분 개수 (2개 이상이면 조각남)
  - a_trab_before/after : 각 ROI로 계산한 10-90% trimmed mean 감쇠값
  - bar_before/after    : 같은 soft/air 기준값으로 계산한 BAR (분모는 동일,
                           trab만 바뀜 -- 순수하게 ROI 제외의 영향만 봄)

QC_MIN_ROI_PX(300px) 미만으로 줄어들면 원래 QC가 이 척추를 실패 처리해야
하므로 별도 표시한다.

사용법 (backend 디렉터리, 모델 필요)
------------------------------------
    python scripts/debug_metal_exclusion_roi_impact.py <dicom 파일 또는 폴더...>
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

_BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from app.services.inference.yolo_engine import CLASS_NAMES, CONF, IMGSZ, YoloBmdEngine

DICOM_EXTS = {".dcm", ".dicom"}
QC_MIN_ROI_PX = 300

# ---- v19 notebook B2, 이식 (프로덕션엔 아직 없음) ----
METAL_SAT_THRESH = 0.99
METAL_FLAT_VAR = 0.0008
METAL_FLAT_WIN = 5
METAL_MIN_BLOB_PX = 12
METAL_HALO_PX = 2


def detect_metal_mask(gray_lin, mask, sat_thresh=METAL_SAT_THRESH, flat_var=METAL_FLAT_VAR,
                       flat_win=METAL_FLAT_WIN, min_blob_px=METAL_MIN_BLOB_PX, halo_px=METAL_HALO_PX):
    m = (mask > 0)
    body_px = int(m.sum())
    if body_px < 20:
        return np.zeros_like(mask, np.uint8), 0.0
    sat = gray_lin >= sat_thresh
    mean = cv2.blur(gray_lin, (flat_win, flat_win))
    mean_sq = cv2.blur(gray_lin.astype(np.float64) ** 2, (flat_win, flat_win))
    local_var = np.clip(mean_sq - mean.astype(np.float64) ** 2, 0, None)
    flat = local_var < flat_var
    cand = (sat & flat & m).astype(np.uint8)
    if not cand.any():
        return np.zeros_like(mask, np.uint8), 0.0
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    core = np.zeros_like(mask, np.uint8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_blob_px:
            core[lbl == i] = 1
    if not core.any():
        return np.zeros_like(mask, np.uint8), 0.0
    k = 2 * halo_px + 1
    halo = cv2.dilate(core, np.ones((k, k), np.uint8))
    out = (halo.astype(bool) & sat & m).astype(np.uint8)
    metal_frac = float(out.sum()) / body_px
    return out, metal_frac


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


def _trimmed_mean(v, lo=10, hi=90):
    if v.size == 0:
        return None
    a, b = np.percentile(v, [lo, hi])
    core = v[(v >= a) & (v <= b)]
    return float(core.mean()) if core.size else float(v.mean())


def analyze(engine: YoloBmdEngine, path: Path):
    model = engine._ensure_model()
    rgb, dens, H, W, *_ = YoloBmdEngine._load_dicom(path)
    response = YoloBmdEngine._detector_response(path)
    gray_lin = YoloBmdEngine._gray_lin(dens)

    r = model.predict(rgb, conf=CONF, imgsz=IMGSZ, verbose=False)[0]
    if r.masks is None or len(r.boxes) == 0:
        return []
    boxes = r.boxes.xyxy.cpu().numpy()
    classes = r.boxes.cls.cpu().numpy().astype(int)
    confs = r.boxes.conf.cpu().numpy()

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
    A = YoloBmdEngine._to_attenuation(gray_lin, response)

    rows = []
    for i in range(len(boxes)):
        label = CLASS_NAMES[classes[i]] if 0 <= classes[i] < len(CLASS_NAMES) else str(classes[i])
        full_mask = YoloBmdEngine._poly_to_mask(r.masks.xy[i], H, W, shrink=1.0)
        trab_roi = YoloBmdEngine._trabecular_roi(full_mask)
        metal_mask, metal_frac = detect_metal_mask(gray_lin, full_mask)
        roi_after = (trab_roi.astype(bool) & ~metal_mask.astype(bool)).astype(np.uint8)

        roi_before_px = int((trab_roi > 0).sum())
        roi_after_px = int((roi_after > 0).sum())
        if roi_before_px == 0:
            continue
        frac_removed = 1.0 - roi_after_px / roi_before_px

        n_frag, _ = cv2.connectedComponents((roi_after > 0).astype(np.uint8), connectivity=8)
        n_fragments = max(0, n_frag - 1)

        crop_box = YoloBmdEngine._crop_box(H, W, boxes[i], pad=0.6)
        bar_before, parts_before = YoloBmdEngine._bar_score(
            gray_lin, trab_roi, valid, crop_box, bone_dil, response, view
        )
        bar_after, parts_after = YoloBmdEngine._bar_score(
            gray_lin, roi_after, valid, crop_box, bone_dil, response, view
        )

        a_trab_before = _trimmed_mean(A[(trab_roi > 0) & valid]) if roi_before_px else None
        a_trab_after = _trimmed_mean(A[(roi_after > 0) & valid]) if roi_after_px else None

        rows.append({
            "file": path.name, "label": label, "metal_frac": metal_frac,
            "roi_before_px": roi_before_px, "roi_after_px": roi_after_px,
            "frac_removed": frac_removed, "n_fragments": n_fragments,
            "a_trab_before": a_trab_before, "a_trab_after": a_trab_after,
            "bar_before": bar_before, "bar_after": bar_after,
            "qc_fail_after": roi_after_px < QC_MIN_ROI_PX,
        })
    return rows


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python scripts/debug_metal_exclusion_roi_impact.py <dicom files or folders...>")
        raise SystemExit(1)

    files = collect_files(sys.argv[1:])
    if not files:
        print("No DICOM files found.")
        raise SystemExit(1)

    engine = YoloBmdEngine()
    all_rows = []
    for f in files:
        try:
            rows = analyze(engine, f)
        except Exception as exc:  # noqa: BLE001
            print(f"{f.name} : [error] {type(exc).__name__}: {exc}")
            continue
        all_rows.extend(rows)

    affected = [r for r in all_rows if r["metal_frac"] > 0]
    affected.sort(key=lambda r: -r["metal_frac"])

    print(f"{'file':38s} {'label':4s} {'m_frac':>6s} {'roi_bef':>8s} {'roi_aft':>8s} "
          f"{'removed':>8s} {'frag':>4s} {'bar_bef':>8s} {'bar_aft':>8s} {'delta':>8s}  flag")
    for r in affected:
        bb = r["bar_before"]
        ba = r["bar_after"]
        bb_s = f"{bb:.3f}" if bb is not None else "None"
        ba_s = f"{ba:.3f}" if ba is not None else "None"
        delta_s = f"{ba-bb:+.3f}" if (bb is not None and ba is not None) else "n/a"
        flag = []
        if r["qc_fail_after"]:
            flag.append("QC_MIN_ROI_PX 미달")
        if r["n_fragments"] >= 2:
            flag.append(f"{r['n_fragments']}조각")
        if bb is not None and ba is not None and abs(ba - bb) > 0.3:
            flag.append("BAR 크게 변함")
        print(
            f"{r['file']:38s} {r['label']:4s} {r['metal_frac']:6.3f} {r['roi_before_px']:8d} "
            f"{r['roi_after_px']:8d} {r['frac_removed']:8.3f} {r['n_fragments']:4d} "
            f"{bb_s:>8s} {ba_s:>8s} {delta_s:>8s}  {' / '.join(flag)}"
        )

    print()
    print(f"금속이 조금이라도 검출된 척추: {len(affected)}건")
    n_qc_fail = sum(1 for r in affected if r["qc_fail_after"])
    n_fragmented = sum(1 for r in affected if r["n_fragments"] >= 2)
    both_bar = [r for r in affected if r["bar_before"] is not None and r["bar_after"] is not None]
    n_big_delta = sum(1 for r in both_bar if abs(r["bar_after"] - r["bar_before"]) > 0.3)
    print(f"  QC_MIN_ROI_PX({QC_MIN_ROI_PX}) 미달로 떨어짐: {n_qc_fail}건")
    print(f"  ROI가 2조각 이상으로 쪼개짐: {n_fragmented}건")
    print(f"  bar_before/after 둘 다 계산됨: {len(both_bar)}건 중 |delta|>0.3: {n_big_delta}건")
    if both_bar:
        deltas = np.array([r["bar_after"] - r["bar_before"] for r in both_bar])
        print(f"  delta(after-before) 분포: median={np.median(deltas):+.3f}  "
              f"p25={np.percentile(deltas,25):+.3f}  p75={np.percentile(deltas,75):+.3f}  "
              f"|max|={np.abs(deltas).max():.3f}")


if __name__ == "__main__":
    main()
