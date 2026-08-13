"""BAR 값 분포 확인 도구 — 히트맵 컬러 스케일(BAR_COLOR_VMIN/VMAX)이 실제
관측되는 BAR 분포와 맞는지 확인한다.

배경
----
"건강한 뼈들인데도 다 파랗게(낮게) 나온다"는 리포트로 나온 진단 도구다.
BAR는 DXA 미교정 proxy라 '정상 범위'가 정의돼 있지 않다 — 지금 고정
스케일(BAR_COLOR_VMIN=0.0, BAR_COLOR_VMAX=1.0)은 이전 대화에서 본 몇 장의
스크린샷 값(0.3~0.9대)을 보고 잡은 추정치일 뿐, 실제 코호트 분포로 검증한
적이 없다. 그 사이 실제로 보인 값 대부분(0.18~0.36대)은 이 스케일에서
파랑~시안 구간에 해당해, 정상 소견도 전부 '낮음'처럼 보이는 원인일 수
있다. 스케일을 또 감으로 바꾸기 전에 실제 분포(percentile)를 먼저 본다
— XAI 이미지/오버레이는 만들지 않고 BAR 숫자만 빠르게 뽑는다
(`YoloBmdEngine._quick_target_bar`, 재현성(LSC) 측정에 쓰는 것과 같은
경량 경로).

사용법 (backend 디렉터리, 모델/torch가 있는 환경)
------------------------------------------------
    python scripts/debug_bar_distribution.py <dicom 파일 또는 폴더...>

예:
    python scripts/debug_bar_distribution.py "C:\\...\\dataset-dcm\\lateral"
    python scripts/debug_bar_distribution.py a.dcm b.dicom "C:\\...\\dataset-dcm"
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


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python scripts/debug_bar_distribution.py <dicom files or folders...>")
        raise SystemExit(1)

    files = collect_files(sys.argv[1:])
    if not files:
        print("No DICOM files found.")
        raise SystemExit(1)

    engine = YoloBmdEngine()
    model = engine._ensure_model()
    rows: list[tuple[str, float, str]] = []
    for f in files:
        try:
            rgb, dens, H, W, *_ = YoloBmdEngine._load_dicom(f)
            response = YoloBmdEngine._detector_response(f)
            view = YoloBmdEngine._view_position(f)
            if view == "unknown":
                # Same fallback as run(): no tag -> shape-based classification.
                r0 = model.predict(rgb, conf=CONF, imgsz=IMGSZ, verbose=False)[0]
                view = (
                    YoloBmdEngine._view_from_shape(
                        r0.boxes.xyxy.cpu().numpy(), r0.masks.xy, H, W
                    )
                    if r0.masks is not None and len(r0.boxes) > 0
                    else "AP"
                )
            bar = engine._quick_target_bar(rgb, dens, response, view)
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] {f.name}: {type(exc).__name__}: {exc}")
            continue
        if bar is None:
            print(f"[skip] {f.name}: BAR unavailable / QC failed")
            continue
        rows.append((f.name, bar, view))
        print(f"{f.name}: BAR={bar:.3f}  view={view}")

    if not rows:
        print("No usable BAR values.")
        return

    arr = np.array([b for _, b, _ in rows])
    print()
    print(f"n = {len(arr)}")
    print(
        f"min={arr.min():.3f}  p5={np.percentile(arr, 5):.3f}  "
        f"p25={np.percentile(arr, 25):.3f}  median={np.percentile(arr, 50):.3f}  "
        f"p75={np.percentile(arr, 75):.3f}  p95={np.percentile(arr, 95):.3f}  "
        f"max={arr.max():.3f}"
    )
    print()
    vmin, vmax = YoloBmdEngine.BAR_COLOR_VMIN, YoloBmdEngine.BAR_COLOR_VMAX
    print(f"현재 히트맵 컬러 스케일: BAR_COLOR_VMIN={vmin}, BAR_COLOR_VMAX={vmax}")
    print(
        f"위 p95/max가 {vmax}보다 한참 낮다면, 그게 대부분의 스터디가 스케일 "
        f"하단(파랑)에 몰려 보이는 이유다 — VMAX를 이 코호트의 p95~max 근처로 "
        "낮추는 게 다음 조정값의 근거가 된다(감으로 다시 잡지 않기 위해)."
    )

    ap_bars = [b for _, b, v in rows if v == "AP"]
    la_bars = [b for _, b, v in rows if v == "LA"]
    if ap_bars and la_bars:
        print()
        print(
            f"AP (n={len(ap_bars)}): median={np.median(ap_bars):.3f}  "
            f"LA (n={len(la_bars)}): median={np.median(la_bars):.3f}"
        )
        print("AP/LA median이 크게 다르면 뷰별로 스케일을 따로 잡는 것도 고려할 만하다.")


if __name__ == "__main__":
    main()
