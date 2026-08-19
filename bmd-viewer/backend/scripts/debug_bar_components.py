"""L4 BAR 계산의 중간값(trab/soft/air/margin)을 파일별로 출력하는 도구.

`debug_bar_distribution.py`는 최종 BAR 스칼라만 뽑지만, air 기준값이
불안정한지("배경이 연부조직/뼈보다 높게 잡히는" 문제 등)를 눈으로 확인하려면
분자·분모를 이루는 개별 감쇠값(a_trab/a_soft/a_air)이 필요하다. 이 스크립트는
`YoloBmdEngine._quick_target_bar`와 동일한 실제 프로덕션 경로
(`_bar_score` 등, 노트북과 별개로 실제 배포되는 코드)를 그대로 타되, 그
과정에서 나오는 a_soft/a_air/margin을 회수하고 a_trab은
`a_trab = bar * margin + a_soft`로 역산해 함께 보여준다(새 ROI 로직을
중복 구현하지 않기 위함 — bar_score 안에서 이미 한 번 계산된 값이므로
역산이 곧 원래 값과 동일하다).

사용법 (backend 디렉터리, 모델/torch가 있는 환경)
------------------------------------------------
    python scripts/debug_bar_components.py <dicom 파일 또는 폴더...>

예:
    python scripts/debug_bar_components.py "C:\\...\\dataset-dcm\\lateral"
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

_BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from app.services.inference.yolo_engine import CONF, IMGSZ, YoloBmdEngine

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


def measure_components(engine: YoloBmdEngine, path: Path):
    """`YoloBmdEngine._quick_target_bar`와 같은 경로를 밟되, a_soft/a_air/margin과
    거기서 역산한 a_trab까지 반환한다. (bar, a_trab, a_soft, a_air, margin, view,
    qc_status) 또는 실패 시 None들."""
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

    bar, qc_status, _msg, _roi, bar_parts = YoloBmdEngine._measure_vertebra_bar(
        gray_lin, dens, r.masks.xy[l4_i], H, W, valid, bone_dil, response, crop_box, view
    )
    if bar is None or "a_soft" not in bar_parts:
        return {
            "bar": None, "a_trab": None, "a_soft": bar_parts.get("a_soft"),
            "a_air": bar_parts.get("a_air"), "margin": bar_parts.get("margin"),
            "view": view, "qc_status": qc_status,
        }

    a_soft = bar_parts["a_soft"]
    a_air = bar_parts["a_air"]
    margin = bar_parts["margin"]
    a_trab = bar * margin + a_soft  # bar_score 내부 계산의 역산 (재구현 아님)
    return {
        "bar": bar, "a_trab": a_trab, "a_soft": a_soft, "a_air": a_air,
        "margin": margin, "view": view, "qc_status": qc_status,
    }


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python scripts/debug_bar_components.py <dicom files or folders...>")
        raise SystemExit(1)

    files = collect_files(sys.argv[1:])
    if not files:
        print("No DICOM files found.")
        raise SystemExit(1)

    engine = YoloBmdEngine()
    n_ok, n_fail = 0, 0
    for f in files:
        try:
            res = measure_components(engine, f)
        except Exception as exc:  # noqa: BLE001
            print(f"{f.name} : [error] {type(exc).__name__}: {exc}")
            n_fail += 1
            continue

        if res is None:
            print(f"{f.name} : [no L4 detected]")
            n_fail += 1
            continue
        if res["bar"] is None:
            print(
                f"{f.name} : soft={res['a_soft']!r} air={res['a_air']!r} "
                f"margin={res['margin']!r} -> BAR unavailable "
                f"(QC={res['qc_status']}, view={res['view']})"
            )
            n_fail += 1
            continue

        print(
            f"{f.name} : {res['a_trab']:.4f} - {res['a_soft']:.4f} "
            f"/ {res['a_soft']:.4f} - {res['a_air']:.4f} = {res['bar']:.4f}"
            f"  (view={res['view']}, QC={res['qc_status']})"
        )
        n_ok += 1

    print()
    print(f"n_ok={n_ok}  n_fail={n_fail}  total={len(files)}")


if __name__ == "__main__":
    main()
