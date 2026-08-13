"""_air_reference 진단 도구 — 계산식은 건드리지 않고, 지금 배경(a_air)으로
채택되는 픽셀이 실제 '직접 노출'인지 '조준(collimation) 차폐 그림자'인지
눈으로 확인한다.

배경 (yolo_engine._air_reference)
----------------------------------
BAR = (a_trab - a_soft) / (a_soft - a_air) 에서 a_air는 "직접 노출(광자가
도달했지만 몸에 감쇠되지 않은) 배경 레벨"이어야 한다. 그런데 현재 구현은
Otsu로 어두운 픽셀을 그냥 '배경'으로 묶어 25 percentile을 쓰는데, 조준
차폐 그림자(광자가 아예 도달하지 않은 영역 — 평탄, 잡음 없음)와 진짜 직접
노출(광자 통계 잡음이 남아 있음)을 구분하지 않는다. 측면(LA) 촬영은 산란선/
피폭 저감을 위해 조준을 좁게 잡는 경우가 많아 배경 대부분이 차폐 그림자인
사례가 흔한데, 이를 a_air로 쓰면 margin(a_soft-a_air)이 부풀어 BAR 전체가
0 쪽으로 눌린다 — "측면 BAR가 이유 없이 낮다"는 리포트로 발견.

이 스크립트는 그 가설을 시각적으로 검증하기 위한 진단 전용 도구다. 국소
표준편차로 배경을 '그림자 후보(평탄)' / '직접노출 후보(잡음 있음)'로
나눠 보여주고, 두 가지 a_air 후보값(현재 방식 vs 직접노출만 썼을 때)을
비교 출력한다. 값 튜닝이나 산식 변경은 하지 않는다 — 확인 후 별도로 반영.

사용법 (backend 디렉터리에서)

(bmd) PS C:\Users\csm02\Desktop\edward\bmd\src\BMD\bmd-viewer\backend> python scripts/debug_air_reference.py "C:\Users\csm02\Desktop\edward\class\Capstone Project INFO-6156\SourceCode\BuddyCheckAI\App\Backend\dcmfiles\lateral\3d819b3ee02c3e1a324fab9b9baa3c6a.dicom" out.png
[3d819b3ee02c3e1a324fab9b9baa3c6a.dicom]
  현재 a_air (otsu bg, 25th pct): 0.0132
  otsu 배경 비율: 62.61%  (그 중 그림자 후보: 81.5%)
  '직접노출 후보만' 썼을 때 a_air: 0.1537  (현재 대비 +0.1404)
  저장: out.png
(bmd) PS C:\Users\csm02\Desktop\edward\bmd\src\BMD\bmd-viewer\backend> python scripts/debug_air_reference.py "C:\Users\csm02\Desktop\edward\class\Capstone Project INFO-6156\SourceCode\BuddyCheckAI\App\Backend\dcmfiles\lateral\50e75f80d4aa28ffbc75ed4770a5ec50.dicom" out1.png
[50e75f80d4aa28ffbc75ed4770a5ec50.dicom]
  현재 a_air (otsu bg, 25th pct): 0.1320
  otsu 배경 비율: 47.20%  (그 중 그림자 후보: 67.2%)
  '직접노출 후보만' 썼을 때 a_air: 0.2214  (현재 대비 +0.0894)
  저장: out1.png
-----------------------------
    python -m scripts.debug_air_reference <dicom_path> [out.png]

여러 장을 한 번에 보고 싶으면 쉘에서 for 문으로 반복 호출하면 된다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

# VS Code의 "Run Python File"처럼 backend/ 밖에서 이 파일을 직접 실행해도
# `app` 패키지를 찾도록, backend/ (이 파일의 조부모 디렉터리)를 sys.path에
# 넣는다. `python -m scripts.debug_air_reference`로 backend/에서 실행할
# 때는 이미 backend/가 sys.path에 있으므로 아무 영향이 없다.
_BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from app.services.inference.yolo_engine import YoloBmdEngine

# 국소(5x5) 표준편차 임계값 (8bit 스케일) — 이 이하면 '광자 잡음이 없다' =
# 차폐 그림자 후보로 본다. 튜닝 전 placeholder, 실측 사례로 눈으로 확인하며
# 조정할 값이다.
LOCAL_STD_THRESHOLD = 2.0


def _label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(out, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


def build_debug_image(dicom_path: Path) -> tuple[np.ndarray, dict]:
    rgb, dens, H, W, *_ = YoloBmdEngine._load_dicom(dicom_path)
    gray = YoloBmdEngine._gray_lin(dens)

    u8 = (np.clip(gray, 0, 1) * 255).astype(np.uint8)
    thr, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bg_mask = u8 < thr
    bg_vals = gray[bg_mask]
    bg_frac = bg_vals.size / gray.size

    # 현재 _air_reference와 완전히 동일한 로직 (건드리지 않음 — 대조용).
    if bg_frac < 0.005:
        current_a_air = float(np.percentile(gray, 1))
        current_path = "fallback: whole-image 1th pct (bg<0.5%)"
    else:
        current_a_air = float(np.percentile(bg_vals, 25))
        current_path = "otsu bg, 25th pct"

    # 배경 안에서만 국소 표준편차로 그림자(평탄) vs 직접노출(잡음) 후보를 나눈다.
    bg_only = np.where(bg_mask, u8, 0).astype(np.float32)
    local_mean = cv2.blur(bg_only, (5, 5))
    local_sq_mean = cv2.blur(bg_only ** 2, (5, 5))
    local_std = np.sqrt(np.clip(local_sq_mean - local_mean ** 2, 0, None))

    shadow_mask = bg_mask & (local_std <= LOCAL_STD_THRESHOLD)
    noisy_mask = bg_mask & (local_std > LOCAL_STD_THRESHOLD)
    shadow_frac_of_bg = shadow_mask.sum() / max(int(bg_mask.sum()), 1)

    if noisy_mask.sum() >= 0.005 * gray.size:
        candidate_a_air = float(np.percentile(gray[noisy_mask], 25))
    else:
        candidate_a_air = None

    # ---- 2x2 진단 패널 ----
    disp_gray_bgr = cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)

    bg_mask_vis = cv2.cvtColor(
        np.where(bg_mask, 255, 0).astype(np.uint8), cv2.COLOR_GRAY2BGR
    )

    std_vis = np.clip(local_std * 20, 0, 255).astype(np.uint8)
    std_vis_bgr = cv2.applyColorMap(std_vis, cv2.COLORMAP_VIRIDIS)
    std_vis_bgr[~bg_mask] = 0

    overlay = disp_gray_bgr.copy()
    overlay[shadow_mask] = (255, 80, 80)   # 파랑(BGR) = 차폐 그림자 후보
    overlay[noisy_mask] = (80, 200, 80)    # 초록 = 직접노출 후보
    blended = cv2.addWeighted(disp_gray_bgr, 0.4, overlay, 0.6, 0)

    panels = [
        _label(disp_gray_bgr, "gray_lin (0.5/99.5pct window)"),
        _label(bg_mask_vis, f"otsu bg mask ({bg_frac * 100:.1f}% of image)"),
        _label(std_vis_bgr, "local std within bg (bright=noisy=likely real air)"),
        _label(blended, "blue=shadow candidate / green=direct-exposure candidate"),
    ]
    grid = np.concatenate(
        [np.concatenate(panels[:2], axis=1), np.concatenate(panels[2:], axis=1)],
        axis=0,
    )

    info = {
        "current_a_air": current_a_air,
        "current_path": current_path,
        "bg_frac": bg_frac,
        "shadow_frac_of_bg": shadow_frac_of_bg,
        "candidate_a_air": candidate_a_air,
    }
    return grid, info


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python -m scripts.debug_air_reference <dicom_path> [out.png]")
        raise SystemExit(1)

    dicom_path = Path(sys.argv[1])
    out_path = (
        Path(sys.argv[2]) if len(sys.argv) > 2
        else dicom_path.with_suffix(".air_debug.png")
    )

    grid, info = build_debug_image(dicom_path)
    cv2.imwrite(str(out_path), grid)

    print(f"[{dicom_path.name}]")
    print(f"  현재 a_air ({info['current_path']}): {info['current_a_air']:.4f}")
    print(f"  otsu 배경 비율: {info['bg_frac'] * 100:.2f}%"
          f"  (그 중 그림자 후보: {info['shadow_frac_of_bg'] * 100:.1f}%)")
    if info["candidate_a_air"] is not None:
        diff = info["candidate_a_air"] - info["current_a_air"]
        print(f"  '직접노출 후보만' 썼을 때 a_air: {info['candidate_a_air']:.4f}"
              f"  (현재 대비 {diff:+.4f})")
    else:
        print("  직접노출로 보이는 배경이 0.5% 미만 -> 이 스터디는 사실상 진짜 "
              "배경이 없음 (가설과 일치: 배경 대부분이 차폐 그림자일 가능성)")
    print(f"  저장: {out_path}")


if __name__ == "__main__":
    main()
