"""한 장의 X-ray에서 air 기준값과 연부조직(soft-tissue) 기준값 영역을 함께
시각화한다 -- `debug_air_visualize.py` + `debug_soft_tissue_visualize.py`를
한 그림에 합친 버전.

모델이 필요하다 (soft-tissue 기준값은 L4 위치가 있어야 계산되므로).

색상
----
  - 노랑   : air 계산에 실제로 쓰인 direct-exposure 픽셀
  - 남색   : air 배경 후보였지만 그림자로 제외된 픽셀
  - 초록   : 연부조직 후보 중 채택된 쪽(더 낮은 값)
  - 하늘색 : 연부조직 후보 중 버려진 쪽
  - 빨강 채움 : 대상(target) 척추(L4) 본체
  - 파랑 윤곽선 : 그 외 검출된 척추

사용법 (backend 디렉터리, 모델 필요)
------------------------------------
    python scripts/debug_air_soft_combined_visualize.py <dicom 파일...> [--out DIR]
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


def _air_regions(gray_lin: np.ndarray):
    """air_reference()와 동일 로직, 시각화용 마스크까지 반환.
    Returns (air_value, direct_exposure_mask, shadow_mask)."""
    u8 = (np.clip(gray_lin, 0, 1) * 255).astype(np.uint8)
    thr, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bg_mask_raw = u8 < thr
    bg_mask = YoloBmdEngine._border_connected_mask(bg_mask_raw)
    bg = gray_lin[bg_mask]
    empty = np.zeros_like(bg_mask)
    if bg.size < 0.005 * gray_lin.size:
        bg_fallback = gray_lin[bg_mask_raw]
        if bg_fallback.size >= 0.005 * gray_lin.size:
            return float(np.percentile(bg_fallback, 25)), bg_mask_raw, empty
        return float(np.percentile(gray_lin, 1)), empty, empty

    bg_only = np.where(bg_mask, u8, 0).astype(np.float32)
    local_mean = cv2.blur(bg_only, (5, 5))
    local_sq_mean = cv2.blur(bg_only ** 2, (5, 5))
    local_std = np.sqrt(np.clip(local_sq_mean - local_mean ** 2, 0, None))
    direct = bg_mask & (local_std > YoloBmdEngine.AIR_SHADOW_STD_MAX)
    if direct.sum() >= 0.005 * gray_lin.size:
        return float(np.percentile(gray_lin[direct], 25)), direct, (bg_mask & ~direct)
    return float(np.percentile(gray_lin[bg_mask], 25)), empty, bg_mask


def visualize_one(engine: YoloBmdEngine, path: Path, out_dir: Path):
    import matplotlib.pyplot as plt

    model = engine._ensure_model()
    rgb, dens, H, W, *_ = YoloBmdEngine._load_dicom(path)
    gray_lin = YoloBmdEngine._gray_lin(dens)

    r = model.predict(rgb, conf=CONF, imgsz=IMGSZ, verbose=False)[0]
    if r.masks is None or len(r.boxes) == 0:
        print(f"[skip] {path.name}: no vertebra detected")
        return
    boxes = r.boxes.xyxy.cpu().numpy()
    classes = r.boxes.cls.cpu().numpy().astype(int)
    confs = r.boxes.conf.cpu().numpy()
    l4_i, _status = YoloBmdEngine._pick_l4_index(boxes, classes, confs)
    if l4_i is None:
        print(f"[skip] {path.name}: L4 not identified")
        return

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
    l4_box = YoloBmdEngine._crop_box(H, W, boxes[l4_i], pad=0.6)

    cx1, cy1, cx2, cy2 = l4_box
    window = np.zeros((H, W), dtype=bool)
    window[cy1:cy2, cx1:cx2] = True
    soft = valid & window & (bone_dil == 0)

    soft_ref = YoloBmdEngine._soft_tissue_ref(gray_lin, soft, valid, l4_box, view)
    air_val, air_direct, air_shadow = _air_regions(gray_lin)

    # 실제 BAR 계산식 -- 화면에 표시된 것과 정확히 같은 프로덕션 경로
    # (_bar_score)로 다시 계산한다. trab_roi는 L4 trabecular ROI.
    response = YoloBmdEngine._detector_response(path)
    l4_mask = YoloBmdEngine._poly_to_mask(r.masks.xy[l4_i], H, W, shrink=1.0)
    trab_roi = YoloBmdEngine._trabecular_roi(l4_mask)
    bar, bar_parts = YoloBmdEngine._bar_score(
        gray_lin, trab_roi, valid, l4_box, bone_dil, response, view
    )
    if bar is not None and "a_soft" in bar_parts:
        a_soft = bar_parts["a_soft"]
        a_air = bar_parts["a_air"]
        margin = bar_parts["margin"]
        a_trab = bar * margin + a_soft  # bar_score 내부 계산의 역산
        formula = (
            f"BAR = (A_trab - A_soft) / (A_soft - A_air) "
            f"= ({a_trab:.4f} - {a_soft:.4f}) / ({a_soft:.4f} - {a_air:.4f}) "
            f"= {bar:.4f}"
        )
    else:
        formula = f"BAR = n/a (soft/air margin unusable: {bar_parts})"

    overlay = cv2.cvtColor((np.clip(gray_lin, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)

    # air: 노랑 = direct-exposure, 남색 = 그림자(제외)
    overlay[air_shadow] = (0.55 * overlay[air_shadow] + 0.45 * np.array([40, 40, 160])).astype(np.uint8)
    overlay[air_direct] = (0.4 * overlay[air_direct] + 0.6 * np.array([255, 230, 40])).astype(np.uint8)

    # soft tissue: 초록 = 채택, 하늘색 = 버려짐
    if view == "AP":
        overlay[soft] = (0.5 * overlay[soft] + 0.5 * np.array([60, 230, 90])).astype(np.uint8)
        soft_detail = "AP: pooled"
    else:
        split_x = (cx1 + cx2) // 2
        left = soft.copy(); left[:, split_x:] = False
        right = soft.copy(); right[:, :split_x] = False
        left_v, right_v = _pct(gray_lin, left), _pct(gray_lin, right)
        chosen_is_left = left_v is not None and (right_v is None or left_v <= right_v)
        chosen, rejected = (left, right) if chosen_is_left else (right, left)
        overlay[rejected] = (0.5 * overlay[rejected] + 0.5 * np.array([120, 200, 255])).astype(np.uint8)
        overlay[chosen] = (0.5 * overlay[chosen] + 0.5 * np.array([60, 230, 90])).astype(np.uint8)
        soft_detail = f"{view}: chosen={'left' if chosen_is_left else 'right'}"

    # 척추 윤곽: L4는 빨강 채움, 나머지는 파랑 윤곽선
    for i in range(len(boxes)):
        m = YoloBmdEngine._poly_to_mask(r.masks.xy[i], H, W, shrink=1.0)
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if i == l4_i:
            fill = overlay.copy()
            cv2.drawContours(fill, cnts, -1, (230, 60, 60), -1)
            overlay = (0.6 * overlay + 0.4 * fill).astype(np.uint8)
            cv2.drawContours(overlay, cnts, -1, (220, 30, 30), 2)
        else:
            cv2.drawContours(overlay, cnts, -1, (80, 160, 255), 1)

    fig, ax = plt.subplots(figsize=(max(5.0, W / H * 7.0), 7.6))
    ax.imshow(overlay)
    ax.axis("off")
    ax.set_title(
        f"{path.name}\n"
        f"air={air_val:.4f} (yellow=used, navy=shadow-excluded)   "
        f"soft={soft_ref:.4f} (green=chosen, skyblue=rejected) [{soft_detail}]\n"
        f"red fill = target(L4), blue outline = other vertebrae",
        fontsize=9,
    )
    fig.text(0.5, 0.01, formula, ha="center", va="bottom", fontsize=10, family="monospace")
    fig.tight_layout(rect=[0, 0.035, 1, 1])
    out_path = out_dir / f"{path.stem}_air_soft.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"{path.name} : air={air_val:.4f}  soft={soft_ref:.4f}  view={view}  {formula}  -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--out", default="figures/air_soft_combined")
    args = parser.parse_args()

    files = collect_files(args.paths)
    if not files:
        print("No DICOM files found.")
        raise SystemExit(1)

    out_dir = (_BACKEND_DIR / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    engine = YoloBmdEngine()
    for f in files:
        try:
            visualize_one(engine, f, out_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"{f.name} : [error] {type(exc).__name__}: {exc}")

    print(f"저장 위치: {out_dir}")


if __name__ == "__main__":
    main()
