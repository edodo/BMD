"""`_soft_tissue_ref()`가 실제로 어느 영역을 가리키는지 파일별로 시각화한다.

air 시각화(`debug_air_visualize.py`)와 같은 목적이지만, 연부조직 기준값은
L4 위치를 알아야 계산되는 값이라(척추 주변의 '고리' 영역) YOLO 모델이
필요하다.

`soft` 후보 영역(= L4 크롭 창 안에서, 유효 조직이면서 어떤 척추 몸통도 아닌
곳)은 `_bar_score()`와 동일하게 구성한다:
    window = L4 crop box(pad=0.6) 안
    soft   = valid & window & (bone_dil == 0)

- **AP**: 고리 전체를 하나로 풀링해서 하위 25%(SOFT_TISSUE_PCT) 분위수를 쓴다.
  → 시각화에서 고리 전체를 한 색으로 표시.
- **LA/unknown**: 좌/우로 나눠 각각 분위수를 구하고 **더 낮은(연부조직에
  가까운) 쪽**을 채택한다(반대쪽은 후관절 등 골구조물로 오염됐을 수 있어서).
  → 시각화에서 채택된 쪽은 초록, 버려진 쪽은 파랑으로 구분.

실제 `_soft_tissue_ref()` 반환값은 그 함수를 그대로 호출해서 얻는다(항상
프로덕션과 일치). 어느 쪽이 채택됐는지 보여주는 로직만 시각화 전용으로
복사했다 — 원본이 바뀌면 이 복사본도 같이 고쳐야 한다.

출력: 파일마다 PNG 한 장, 파일명 맨 앞에 soft 값을 붙인다.

사용법 (backend 디렉터리, 모델/torch가 있는 환경)
------------------------------------------------
    python scripts/debug_soft_tissue_visualize.py <dicom 파일 또는 폴더...> [--out DIR] [--limit N]
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


def _pct(gray_lin: np.ndarray, mask: np.ndarray) -> float | None:
    vals = gray_lin[mask]
    return float(np.percentile(vals, YoloBmdEngine.SOFT_TISSUE_PCT)) if vals.size >= 50 else None


def visualize_one(engine: YoloBmdEngine, path: Path, out_dir: Path):
    import matplotlib.pyplot as plt

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
    l4_box = YoloBmdEngine._crop_box(H, W, boxes[l4_i], pad=0.6)

    cx1, cy1, cx2, cy2 = l4_box
    window = np.zeros((H, W), dtype=bool)
    window[cy1:cy2, cx1:cx2] = True
    soft = valid & window & (bone_dil == 0)

    official_ref = YoloBmdEngine._soft_tissue_ref(gray_lin, soft, valid, l4_box, view)

    overlay = cv2.cvtColor((np.clip(gray_lin, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    # 제외된 척추 몸통(bone_dil) 윤곽선 -- 고리가 뼈를 피해가는 걸 눈으로 확인.
    cnts, _ = cv2.findContours(bone_dil.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(overlay, cnts, -1, (255, 255, 255), 1)

    if view == "AP":
        overlay[soft] = (0.5 * overlay[soft] + 0.5 * np.array([60, 230, 90])).astype(np.uint8)
        detail = "AP: ring pooled (green)"
    else:
        split_x = (cx1 + cx2) // 2
        left = soft.copy(); left[:, split_x:] = False
        right = soft.copy(); right[:, :split_x] = False
        left_v, right_v = _pct(gray_lin, left), _pct(gray_lin, right)
        # 더 낮은 쪽 채택 -- _soft_tissue_ref와 동일 로직
        chosen_is_left = (
            left_v is not None and (right_v is None or left_v <= right_v)
        )
        chosen, rejected = (left, right) if chosen_is_left else (right, left)
        overlay[rejected] = (0.5 * overlay[rejected] + 0.5 * np.array([60, 90, 255])).astype(np.uint8)
        overlay[chosen] = (0.5 * overlay[chosen] + 0.5 * np.array([60, 230, 90])).astype(np.uint8)
        detail = (
            f"{view}: left={left_v!r} right={right_v!r} -> "
            f"chosen={'left' if chosen_is_left else 'right'} (green)"
        )

    fig, ax = plt.subplots(figsize=(max(4.0, W / H * 5.0), 5.0))
    ax.imshow(overlay)
    ax.axis("off")
    ref_str = f"{official_ref:.4f}" if official_ref is not None else "None"
    ax.set_title(
        f"{path.name}\nsoft={ref_str}  |  {detail}\n"
        "green = used (chosen side)   blue = candidate but rejected   white = excluded vertebra outline",
        fontsize=8,
    )
    fig.tight_layout()
    fname_val = official_ref if official_ref is not None else -1.0
    out_path = out_dir / f"{fname_val:.4f}_{path.stem}.png"
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return official_ref, view


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--out", default="figures/soft_viz")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    files = collect_files(args.paths)
    if args.limit:
        files = files[: args.limit]
    if not files:
        print("No DICOM files found.")
        raise SystemExit(1)

    out_dir = (_BACKEND_DIR / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    engine = YoloBmdEngine()
    n_ok = 0
    for i, f in enumerate(files, 1):
        try:
            res = visualize_one(engine, f, out_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] {f.name}: {type(exc).__name__}: {exc}")
            continue
        if res is None:
            print(f"[skip] {f.name}: no L4 detected")
            continue
        n_ok += 1
        if i % 20 == 0 or i == len(files):
            print(f"  ...{i}/{len(files)}")

    print()
    print(f"저장 위치: {out_dir}  ({n_ok}/{len(files)}장 성공)")


if __name__ == "__main__":
    main()
