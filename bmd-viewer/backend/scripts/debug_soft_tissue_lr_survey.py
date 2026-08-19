"""LA(측면) 뷰에서 연부조직 기준값의 좌/우 후보 값 차이를 전수 조사한다.

`debug_soft_tissue_visualize.py`로 몇 장을 눈으로 보니, 더 낮은 쪽을 채택하는
로직 자체는 맞지만 좌우 차이가 거의 없는(사실상 어느 쪽을 골라도 그만인)
케이스가 섞여 있었다. 이게 얼마나 흔한지, 그리고 육안으로 안 본 나머지
파일들 중에 더 이상한 케이스(둘 다 표본 부족, 차이가 극단적으로 큰 경우 등)가
있는지 전수 확인한다.

측정 (모델 필요 -- L4 위치가 있어야 soft-tissue 창을 만들 수 있음)
-------------------------------------------------------------
  - left / right : 각 후보의 SOFT_TISSUE_PCT 분위수 (표본<50px면 None)
  - diff         : |left - right| (둘 다 있을 때만)
  - chosen       : 실제로 채택된 쪽(더 낮은 값)
  - ref          : `_soft_tissue_ref()`가 실제로 반환하는 최종값(그대로 호출,
                   프로덕션과 항상 일치)

AP 이미지는 좌우를 안 나누고 통째로 쓰므로 이 조사에서 제외한다(diff 개념이
없음) -- 뷰별 개수는 요약에 같이 찍는다.

사용법
------
    python scripts/debug_soft_tissue_lr_survey.py <dicom 파일 또는 폴더...>
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
DIFF_FLAG_THRESHOLD = 0.02  # 이보다 좁으면 "사실상 동전 던지기"로 표시


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


def _pct(gray_lin: np.ndarray, mask: np.ndarray) -> float | None:
    vals = gray_lin[mask]
    return float(np.percentile(vals, YoloBmdEngine.SOFT_TISSUE_PCT)) if vals.size >= 50 else None


def analyze(engine: YoloBmdEngine, path: Path):
    model = engine._ensure_model()
    rgb, dens, H, W, *_ = YoloBmdEngine._load_dicom(path)

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
    l4_box = YoloBmdEngine._crop_box(H, W, boxes[l4_i], pad=0.6)

    cx1, cy1, cx2, cy2 = l4_box
    window = np.zeros((H, W), dtype=bool)
    window[cy1:cy2, cx1:cx2] = True
    soft = valid & window & (bone_dil == 0)

    ref = YoloBmdEngine._soft_tissue_ref(gray_lin, soft, valid, l4_box, view)

    if view == "AP":
        return {"view": view, "ref": ref, "left": None, "right": None, "diff": None}

    split_x = (cx1 + cx2) // 2
    left = soft.copy(); left[:, split_x:] = False
    right = soft.copy(); right[:, :split_x] = False
    left_v, right_v = _pct(gray_lin, left), _pct(gray_lin, right)
    diff = abs(left_v - right_v) if (left_v is not None and right_v is not None) else None
    return {"view": view, "ref": ref, "left": left_v, "right": right_v, "diff": diff}


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python scripts/debug_soft_tissue_lr_survey.py <dicom files or folders...>")
        raise SystemExit(1)

    files = collect_files(sys.argv[1:])
    if not files:
        print("No DICOM files found.")
        raise SystemExit(1)

    engine = YoloBmdEngine()
    la_rows: list[dict] = []
    ap_rows: list[dict] = []
    n_no_l4 = 0
    n_no_ref = 0
    n_one_side_only = 0
    for i, f in enumerate(files, 1):
        try:
            res = analyze(engine, f)
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] {f.name}: {type(exc).__name__}: {exc}")
            continue
        if res is None:
            n_no_l4 += 1
            print(f"{f.name} : [no L4 detected]")
            continue
        if res["ref"] is None:
            n_no_ref += 1
        if res["view"] == "AP":
            ap_rows.append(res)
            continue
        row = {"file": f.name, **res}
        la_rows.append(row)
        if res["diff"] is None and not (res["left"] is None and res["right"] is None):
            n_one_side_only += 1
        if i % 20 == 0 or i == len(files):
            print(f"  ...{i}/{len(files)}")

    print()
    print(f"n_total={len(files)}  no_l4_detected={n_no_l4}  AP={len(ap_rows)}  LA/unknown={len(la_rows)}")
    print(f"soft_tissue_ref()가 None인 케이스: {n_no_ref}건 (표본 부족 등으로 값 자체를 못 낸 경우)")
    print(f"좌/우 한쪽만 표본 충분(diff 계산 불가): {n_one_side_only}건")
    print()

    with_diff = [r for r in la_rows if r["diff"] is not None]
    if with_diff:
        with_diff.sort(key=lambda r: r["diff"])
        print(f"{'file':45s} {'left':>7s} {'right':>7s} {'diff':>7s} {'ref':>7s}  flag")
        for r in with_diff:
            flag = "<<< 사실상 동전 던지기" if r["diff"] < DIFF_FLAG_THRESHOLD else ""
            print(
                f"{r['file']:45s} {r['left']:7.4f} {r['right']:7.4f} {r['diff']:7.4f} "
                f"{r['ref']:7.4f}  {flag}"
            )

        diffs = np.array([r["diff"] for r in with_diff])
        n_small = int((diffs < DIFF_FLAG_THRESHOLD).sum())
        print()
        print(
            f"diff 분포: median={np.median(diffs):.4f}  p10={np.percentile(diffs,10):.4f}  "
            f"p25={np.percentile(diffs,25):.4f}  p75={np.percentile(diffs,75):.4f}  max={diffs.max():.4f}"
        )
        print(
            f"diff < {DIFF_FLAG_THRESHOLD} (사실상 동전 던지기): {n_small}/{len(with_diff)}건 "
            f"({100*n_small/len(with_diff):.1f}%)"
        )
    else:
        print("좌/우 둘 다 표본이 충분한(diff 계산 가능한) LA 케이스가 없습니다.")


if __name__ == "__main__":
    main()
