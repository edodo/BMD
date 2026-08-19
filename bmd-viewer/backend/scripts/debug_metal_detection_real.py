"""B2(`detect_metal_mask`, v19 notebook)를 실제 이물질(핀) DICOM으로 검증한다.

배경
----
v19 노트북 B2 섹션은 지금까지 synthetic(이미지에 인위적으로 그려 넣은 포화
사각형)으로만 검증됐다 -- 노트북 자신도 "neither BUU-LSPINE-2000 nor the
42-study test set has a confirmed post-surgical/instrumented case, so this
detector cannot be validated against a real positive in this cohort yet"라고
그 공백을 명시했다. 사용자가 실제 이물질(핀) DICOM 위치를 제공해서 그 공백을
메운다.

또한 `detect_metal_mask()`는 아직 프로덕션 백엔드(yolo_engine.py)에 이식되지
않았다 -- 백엔드는 여전히 예전 방식(`bmd_metal_ratio`, 유효 조직 p95 배수
기반)을 쓰는데, 이건 2026-06 실측 검증([[bmd-normalization-method]] 메모)에서
이미 "실데이터에서 무력함"이 확인된 방식이다. 그래서 아래 `detect_metal_mask()`
는 v19 노트북 B2 셀에서 그대로 복사했다 -- 백엔드에 아직 없는 함수라 import가
아니라 이식(포팅)이다. 원본이 바뀌면 이 복사본도 같이 고쳐야 한다.

측정
----
파일마다 검출된 모든 척추(L1~L5)에 대해 metal_frac을 계산하고, 가장 높은
척추를 "핀으로 추정되는 위치"로 표시한다. `QC_METAL_WARN_FRAC`(0.10, v19
config와 동일 값)을 넘으면 "감지됨"으로 집계한다. 파일마다 시각화 PNG도
저장한다(빨강 = detect_metal_mask() 출력, 흰 윤곽선 = 척추 마스크).

사용법 (backend 디렉터리, 모델 필요)
------------------------------------
    python scripts/debug_metal_detection_real.py <dicom 파일 또는 폴더...> [--out DIR]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

_BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from app.services.inference.yolo_engine import CLASS_NAMES, CONF, IMGSZ, YoloBmdEngine

DICOM_EXTS = {".dcm", ".dicom"}
QC_METAL_WARN_FRAC = 0.10  # v19 notebook config와 동일 값 -- 이 이상이면 "감지됨"

# ---- v19 notebook B2 cell, 그대로 이식 (프로덕션 백엔드엔 아직 없는 함수) ----
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


def analyze(engine: YoloBmdEngine, path: Path):
    model = engine._ensure_model()
    rgb, dens, H, W, *_ = YoloBmdEngine._load_dicom(path)
    gray_lin = YoloBmdEngine._gray_lin(dens)

    r = model.predict(rgb, conf=CONF, imgsz=IMGSZ, verbose=False)[0]
    if r.masks is None or len(r.boxes) == 0:
        return None
    classes = r.boxes.cls.cpu().numpy().astype(int)
    confs = r.boxes.conf.cpu().numpy()

    rows = []
    masks_by_label = {}
    for i in range(len(classes)):
        m = YoloBmdEngine._poly_to_mask(r.masks.xy[i], H, W, shrink=1.0)
        mm, mf = detect_metal_mask(gray_lin, m)
        label = CLASS_NAMES[classes[i]] if 0 <= classes[i] < len(CLASS_NAMES) else str(classes[i])
        rows.append({"label": label, "metal_frac": mf, "conf": float(confs[i])})
        masks_by_label[label] = (m, mm)
    return gray_lin, rows, masks_by_label


def visualize(gray_lin, rows, masks_by_label, path: Path, out_dir: Path):
    import matplotlib.pyplot as plt

    best = max(rows, key=lambda r: r["metal_frac"]) if rows else None
    overlay = cv2.cvtColor((np.clip(gray_lin, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    for _label, (m, mm) in masks_by_label.items():
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(overlay, cnts, -1, (255, 255, 255), 1)
        if mm.any():
            mmb = mm.astype(bool)
            overlay[mmb] = (0.3 * overlay[mmb] + 0.7 * np.array([255, 40, 40])).astype(np.uint8)

    fig, ax = plt.subplots(figsize=(6, 8))
    ax.imshow(overlay)
    ax.axis("off")
    title_frac = f"{best['label']}: metal_frac={best['metal_frac']:.3f}" if best else "no vertebra detected"
    ax.set_title(
        f"{path.name}\n{title_frac}\nred = detect_metal_mask() output, white = vertebra outline",
        fontsize=9,
    )
    fig.tight_layout()
    fname_val = best["metal_frac"] if best else -1.0
    out_path = out_dir / f"{fname_val:.4f}_{path.stem}.png"
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--out", default="figures/metal_viz")
    args = parser.parse_args()

    files = collect_files(args.paths)
    if not files:
        print("No DICOM files found.")
        raise SystemExit(1)

    out_dir = (_BACKEND_DIR / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    engine = YoloBmdEngine()
    n_detected = 0
    n_no_vert = 0
    max_fracs = []
    for f in files:
        try:
            res = analyze(engine, f)
        except Exception as exc:  # noqa: BLE001
            print(f"{f.name} : [error] {type(exc).__name__}: {exc}")
            continue
        if res is None:
            n_no_vert += 1
            print(f"{f.name} : [no vertebra detected]")
            continue
        gray_lin, rows, masks_by_label = res
        rows_sorted = sorted(rows, key=lambda r: -r["metal_frac"])
        best = rows_sorted[0]
        max_fracs.append(best["metal_frac"])
        flagged = best["metal_frac"] >= QC_METAL_WARN_FRAC
        if flagged:
            n_detected += 1
        detail = "  ".join(f"{r['label']}={r['metal_frac']:.3f}" for r in rows_sorted)
        flag = " <<< metal detected" if flagged else ""
        print(f"{f.name} : best={best['label']} frac={best['metal_frac']:.3f}{flag}   [{detail}]")

        try:
            visualize(gray_lin, rows, masks_by_label, f, out_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"  [viz failed] {type(exc).__name__}: {exc}")

    print()
    print(f"n_total={len(files)}  no_vertebra_detected={n_no_vert}")
    print(
        f"metal_frac >= {QC_METAL_WARN_FRAC}(QC_METAL_WARN_FRAC, 감지됨으로 판정): "
        f"{n_detected}/{len(files)}건 ({100*n_detected/len(files):.1f}%)"
    )
    if max_fracs:
        arr = np.array(max_fracs)
        print(
            f"파일별 최고 metal_frac: median={np.median(arr):.3f}  p25={np.percentile(arr,25):.3f}  "
            f"p75={np.percentile(arr,75):.3f}  max={arr.max():.3f}"
        )
    print(f"저장 위치: {out_dir}")


if __name__ == "__main__":
    main()
