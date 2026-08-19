"""air 기준값이 '몸통 내부까지 깊게 파고든' 오염 사례가 코호트 전체에서
얼마나 흔한지 전수 조사한다.

배경
----
`debug_air_visualize.py`로 몇 장을 눈으로 보고, 초록(direct-exposure) 영역이
몸통 실루엣 안쪽(장내가스·폐야)까지 파고드는 파일들을 발견했다. 테두리-연결성
필터(`_border_connected_mask`, 현재 프로덕션에 반영됨)로 "테두리에 안 닿은
고립된 섬"은 걸러지지만, 테두리에서 시작해 이미지 안쪽 깊숙이까지 이어지는
거대한 덩어리는 못 걸러낸다는 것도 확인됐다. "테두리에서 일정 거리 이내로
제한" 방식은 이런 나쁜 케이스는 개선했지만 원래 깨끗했던 케이스(배경이 넓고
진짜 깨끗한 파일들)를 오히려 악화시켜 폐기했다.

다음 결정(정교한 해부학 기반 필터에 투자할지, QC 플래그로 충분한지)을 내리기
전에 이 문제가 코호트에서 얼마나 흔한지 데이터로 본다.

v2: 첫 버전은 "가장 깊은 픽셀 하나"(max_depth)를 지표로 썼는데, 176개 전부
0.3을 넘고(중앙값 0.921) air 값과 상관관계도 0.064로 사실상 없었다 — 잡음
픽셀 하나, 디스크 틈 하나가 우연히 조건을 통과해도 그 하나 때문에 지표가
포화돼버려 "얼마나 오염됐는지"를 전혀 못 쟀다는 뜻이다. 그래서 단일 최댓값
대신 **"실제 air 값 계산에 쓰인 픽셀들 중 몇 %가 안쪽(깊은 곳)에 있는가"**
(비율, `deep_frac`)로 바꿨다 — 픽셀 한두 개짜리 잡음에는 거의 흔들리지 않고,
정말 넓은 영역이 몸통 내부로 파고들었을 때만 커진다.

측정하는 것 (모델 불필요, DICOM 로드 + Otsu만)
--------------------------------------------
  - bg_frac    : 테두리-연결 Otsu 배경이 전체 이미지에서 차지하는 비율
  - direct_frac: 그중 실제 air 값 계산에 쓰인 direct-exposure 비율
  - deep_frac  : 그 direct-exposure(또는 폴백) 픽셀들 중, 테두리에서
                 DEEP_CUTOFF_FRAC(기본 0.15, 짧은 변의 15%)보다 더 깊은
                 곳에 있는 픽셀의 비율. 이게 큰 파일이 "몸통 내부까지
                 파고든" 의심 사례.

사용법
------
    python scripts/debug_air_contamination_survey.py <dicom 파일 또는 폴더...>
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

_BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from app.services.inference.yolo_engine import YoloBmdEngine

DICOM_EXTS = {".dcm", ".dicom"}
DEEP_CUTOFF_FRAC = 0.15  # 짧은 변 대비 이 비율보다 테두리에서 멀면 "깊다"
DEEP_FLAG_THRESHOLD = 0.30  # deep_frac이 이보다 크면 "의심" 표시


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


def analyze(path: Path):
    """(air, bg_frac, direct_frac, deep_frac, tier) 또는 실패 시 None."""
    _rgb, dens, H, W, *_ = YoloBmdEngine._load_dicom(path)
    gray = YoloBmdEngine._gray_lin(dens)

    u8 = (np.clip(gray, 0, 1) * 255).astype(np.uint8)
    thr, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bg_mask_raw = u8 < thr
    bg_mask = YoloBmdEngine._border_connected_mask(bg_mask_raw)
    bg = gray[bg_mask]

    ys, xs = np.indices((H, W))
    border_dist = np.minimum.reduce([ys, H - 1 - ys, xs, W - 1 - xs]).astype(np.float32)
    short_side = max(1.0, float(min(H, W)))
    is_deep = (border_dist / short_side) > DEEP_CUTOFF_FRAC

    def _deep_frac(mask: np.ndarray) -> float:
        n = int(mask.sum())
        return float((mask & is_deep).sum() / n) if n else 0.0

    if bg.size < 0.005 * gray.size:
        bg_fallback = gray[bg_mask_raw]
        if bg_fallback.size >= 0.005 * gray.size:
            air = float(np.percentile(bg_fallback, 25))
            return air, float(bg_mask_raw.mean()), float(bg_mask_raw.mean()), _deep_frac(bg_mask_raw), "tier2b"
        air = float(np.percentile(gray, 1))
        return air, 0.0, 0.0, 0.0, "tier3"

    bg_only = np.where(bg_mask, u8, 0).astype(np.float32)
    local_mean = cv2.blur(bg_only, (5, 5))
    local_sq_mean = cv2.blur(bg_only ** 2, (5, 5))
    local_std = np.sqrt(np.clip(local_sq_mean - local_mean ** 2, 0, None))
    direct = bg_mask & (local_std > YoloBmdEngine.AIR_SHADOW_STD_MAX)

    if direct.sum() >= 0.005 * gray.size:
        air = float(np.percentile(gray[direct], 25))
        return air, float(bg_mask.mean()), float(direct.mean()), _deep_frac(direct), "tier1"

    air = float(np.percentile(bg, 25))
    return air, float(bg_mask.mean()), 0.0, _deep_frac(bg_mask), "tier2"


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python scripts/debug_air_contamination_survey.py <dicom files or folders...>")
        raise SystemExit(1)

    files = collect_files(sys.argv[1:])
    if not files:
        print("No DICOM files found.")
        raise SystemExit(1)

    rows: list[tuple[str, float, float, float, float, str]] = []
    for f in files:
        try:
            air, bg_frac, direct_frac, deep_frac, tier = analyze(f)
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] {f.name}: {type(exc).__name__}: {exc}")
            continue
        rows.append((f.name, air, bg_frac, direct_frac, deep_frac, tier))

    if not rows:
        print("No usable results.")
        return

    rows.sort(key=lambda r: -r[4])  # deep_frac 내림차순 -> 의심 사례가 위로
    print(f"{'file':45s} {'air':>7s} {'bg_frac':>8s} {'direct_frac':>11s} {'deep_frac':>9s}  tier  flag")
    n_flag = 0
    for name, air, bg_frac, direct_frac, deep_frac, tier in rows:
        flag = "<<<" if deep_frac >= DEEP_FLAG_THRESHOLD else ""
        if flag:
            n_flag += 1
        print(
            f"{name:45s} {air:7.4f} {bg_frac:8.3f} {direct_frac:11.3f} {deep_frac:9.3f}  {tier:6s} {flag}"
        )

    deeps = np.array([r[4] for r in rows])
    airs = np.array([r[1] for r in rows])
    print()
    print(f"n={len(rows)}  |  deep_frac >= {DEEP_FLAG_THRESHOLD} (의심): {n_flag}건 "
          f"({100*n_flag/len(rows):.1f}%)")
    print(
        f"deep_frac: median={np.median(deeps):.3f}  p75={np.percentile(deeps,75):.3f}  "
        f"p90={np.percentile(deeps,90):.3f}  max={deeps.max():.3f}"
    )
    print(
        f"air:       median={np.median(airs):.3f}  p75={np.percentile(airs,75):.3f}  "
        f"p90={np.percentile(airs,90):.3f}  max={airs.max():.3f}"
    )
    if len(rows) >= 3:
        corr = float(np.corrcoef(deeps, airs)[0, 1])
        print(f"corr(deep_frac, air) = {corr:.3f}  (양의 상관이 크면 '내부로 깊이 파고든 비율이 "
              "높을수록 air가 튄다' 가설 지지)")


if __name__ == "__main__":
    main()
