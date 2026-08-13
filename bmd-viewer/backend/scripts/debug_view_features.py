"""Diagnostic tool: image shape features for AP/LA view classification.

Background
----------
v18_spec.md Sec.4.5 specifies view detection as "DICOM ViewPosition/
SeriesDescription tag first, then an aspect-ratio/shape heuristic if
absent," but the actual implementation (`YoloBmdEngine._view_position`)
skipped the second tier and just defaulted to "AP" when the tag was
missing. Since this cohort's DICOM anonymization strips ViewPosition, that
default was silently mis-registering every genuinely-lateral study as
"AP" (user-confirmed: "every file in the lateral folder really is
lateral").

Instead of guessing the shape heuristic, this script measures what
geometric features actually separate AP from LA on two cohorts whose
labels are already known (from the DICOM tag, or user-confirmed by
folder) — same principle as calibrating the BAR color scale
(`debug_bar_distribution.py`): look at the measured distribution before
picking a threshold.

Features measured (all computed from the YOLO vertebra segmentation, no
DICOM tag needed)
------------------------------------------------------------------------
- img_ar        : overall image aspect ratio (W/H)
- bone_ar       : bounding-box aspect ratio (W/H) of the union of all
                  detected vertebra masks
- vert_ar_mean  : mean per-vertebra bounding-box aspect ratio
- symmetry_iou  : IoU of the combined vertebra mask against its own
                  left-right mirror, cropped to its bounding box. AP is
                  assumed to score high (bilateral symmetry — paired
                  pedicles, etc.) and LA low (asymmetric due to posterior
                  elements) — this script tests that assumption directly.

Usage (from the backend directory, in an environment with the model/torch)
----------------------------------------------------------------------------
    python scripts/debug_view_features.py <label>=<dicom folder> [<label>=<folder> ...]

Example:
    python scripts/debug_view_features.py AP="C:\\...\\dataset-dcm\\ap" LA="C:\\...\\dataset-dcm\\lateral"
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from app.services.inference.yolo_engine import CONF, IMGSZ, YoloBmdEngine

DICOM_EXTS = {".dcm", ".dicom"}


def collect_files(folder: str) -> list[Path]:
    p = Path(folder)
    if p.is_file():
        return [p]
    return sorted(
        f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in DICOM_EXTS
    )


def compute_features(dicom_path: Path, model) -> dict | None:
    rgb, dens, H, W, *_ = YoloBmdEngine._load_dicom(dicom_path)
    r = model.predict(rgb, conf=CONF, imgsz=IMGSZ, verbose=False)[0]
    if r.masks is None or len(r.boxes) == 0:
        return None

    boxes = r.boxes.xyxy.cpu().numpy()
    n = len(boxes)

    bone_union = np.zeros((H, W), np.uint8)
    vert_ars = []
    for i in range(n):
        m = YoloBmdEngine._poly_to_mask(r.masks.xy[i], H, W, shrink=1.0)
        bone_union |= m
        bw = float(boxes[i][2] - boxes[i][0])
        bh = float(boxes[i][3] - boxes[i][1])
        if bh > 0:
            vert_ars.append(bw / bh)

    ys, xs = np.where(bone_union > 0)
    if xs.size == 0:
        return None
    bx0, bx1, by0, by1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    bone_w, bone_h = bx1 - bx0 + 1, by1 - by0 + 1
    bone_ar = bone_w / bone_h if bone_h > 0 else float("nan")

    crop = bone_union[by0 : by1 + 1, bx0 : bx1 + 1] > 0
    flipped = crop[:, ::-1]
    inter = np.logical_and(crop, flipped).sum()
    union = np.logical_or(crop, flipped).sum()
    symmetry_iou = float(inter) / float(union) if union > 0 else float("nan")

    return {
        "n_vert": n,
        "img_ar": W / H,
        "bone_ar": bone_ar,
        "vert_ar_mean": float(np.mean(vert_ars)) if vert_ars else float("nan"),
        "symmetry_iou": symmetry_iou,
    }


def print_summary(label: str, rows: list[dict]) -> None:
    print(f"\n[{label}] n = {len(rows)}")
    if not rows:
        return
    for key in ("img_ar", "bone_ar", "vert_ar_mean", "symmetry_iou"):
        arr = np.array([r[key] for r in rows if np.isfinite(r[key])])
        if arr.size == 0:
            continue
        print(
            f"  {key:14s} median={np.median(arr):.3f}  "
            f"p25={np.percentile(arr, 25):.3f}  p75={np.percentile(arr, 75):.3f}  "
            f"min={arr.min():.3f}  max={arr.max():.3f}"
        )


def main() -> None:
    if len(sys.argv) < 2 or "=" not in sys.argv[1]:
        print(
            "usage: python scripts/debug_view_features.py "
            "<label>=<dicom folder> [<label>=<folder> ...]"
        )
        raise SystemExit(1)

    engine = YoloBmdEngine()
    model = engine._ensure_model()

    groups: dict[str, list[dict]] = {}
    for arg in sys.argv[1:]:
        label, _, folder = arg.partition("=")
        files = collect_files(folder)
        rows: list[dict] = []
        for f in files:
            try:
                feats = compute_features(f, model)
            except Exception as exc:  # noqa: BLE001
                print(f"[skip] {f.name}: {type(exc).__name__}: {exc}")
                continue
            if feats is None:
                print(f"[skip] {f.name}: no vertebrae detected")
                continue
            rows.append(feats)
            print(
                f"{f.name}: n_vert={feats['n_vert']} img_ar={feats['img_ar']:.3f} "
                f"bone_ar={feats['bone_ar']:.3f} vert_ar_mean={feats['vert_ar_mean']:.3f} "
                f"symmetry_iou={feats['symmetry_iou']:.3f}"
            )
        groups[label] = rows

    for label, rows in groups.items():
        print_summary(label, rows)


if __name__ == "__main__":
    main()
