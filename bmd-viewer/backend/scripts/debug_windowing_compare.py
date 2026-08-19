"""air 계산의 입력을 만들 때, 이미지별 0.5~99.5 percentile 대비 재조정 대신
DICOM의 고정 스케일(WindowCenter/Width 또는 BitsStored)을 쓰면 어떻게
달라지는지 비교한다.

배경
----
`YoloBmdEngine._load_dicom()`은 매 이미지마다 자기 자신의 0.5~99.5
percentile로 8bit 대비를 다시 늘린다(`_gray_lin`도 동일 원리, dens 기준으로
한 번 더). BAR 자체의 수식(분자·분모가 둘 다 차이)은 이 재조정에 수학적으로
불변이지만, **Otsu가 배경(air)을 가르는 임계값은 이 재조정된 히스토그램
모양에 의존**한다 -- 그래서 골반처럼 아주 밝은 게 크게 잡힌 이미지는 그
밝은 영역이 상위 percentile을 차지해, 배경 판정 자체가 이미지마다 흔들릴 수
있다는 게 지금까지의 결론이었다.

이 스크립트는 그 재조정을 걷어내고, 대신 **이미지 내용과 무관한 고정
스케일**(DICOM `WindowCenter`/`WindowWidth`가 있으면 그걸, 없으면
`BitsStored` 기준 전체 범위)로 8bit 이미지를 만들어 같은 air_reference
로직(Otsu + 테두리-연결성 + 그림자/직접노출 구분)을 돌린 뒤, 두 방식의
air 값과 deep_frac(내부 오염 의심 비율, debug_air_contamination_survey.py와
동일 정의)을 나란히 비교한다.

모델 불필요.

사용법
------
    python scripts/debug_windowing_compare.py <dicom 파일 또는 폴더...>
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pydicom

_BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from app.services.inference.yolo_engine import YoloBmdEngine

DICOM_EXTS = {".dcm", ".dicom"}
DEEP_CUTOFF_FRAC = 0.15


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


def _load_dens(path: Path):
    """YoloBmdEngine._load_dicom()의 dens 계산과 동일 (RescaleSlope/Intercept +
    MONOCHROME1 방향 정규화만, percentile 대비 재조정은 아직 안 함)."""
    ds = pydicom.dcmread(str(path))
    raw = ds.pixel_array.astype(np.float32)
    raw = raw * float(getattr(ds, "RescaleSlope", 1.0)) + float(
        getattr(ds, "RescaleIntercept", 0.0)
    )
    mono1 = str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1"
    dens = (raw.max() - raw) if mono1 else raw.copy()
    return ds, dens, mono1


def _fixed_scale_bounds(ds, dens: np.ndarray, mono1: bool):
    """DICOM WindowCenter/Width(있으면) 또는 BitsStored 기준 고정 [lo, hi].
    dens는 이미 mono1 방향 정규화가 끝난 배열이므로, WindowCenter/Width는
    raw 방향 기준이라 mono1이면 뒤집어서 dens 방향으로 맞춘다."""
    wc = getattr(ds, "WindowCenter", None)
    ww = getattr(ds, "WindowWidth", None)
    if wc is not None and ww is not None:
        wc = float(wc[0]) if hasattr(wc, "__iter__") else float(wc)
        ww = float(ww[0]) if hasattr(ww, "__iter__") else float(ww)
        raw_lo, raw_hi = wc - ww / 2.0, wc + ww / 2.0
        if mono1:
            raw_max = float(dens.max() + raw_lo)  # dens = raw.max() - raw -> raw.max() 복원 근사 불가
            # mono1 방향 전환은 raw.max()가 필요한데 여기선 dens만 있으므로,
            # 근사 대신 아예 raw 자체를 다시 읽지 않고 태그 기반 상한을 쓴다.
            bits = int(getattr(ds, "BitsStored", 16))
            raw_full_max = float(2**bits - 1)
            lo, hi = raw_full_max - raw_hi, raw_full_max - raw_lo
        else:
            lo, hi = raw_lo, raw_hi
        source = f"WindowCenter/Width({wc:.0f}/{ww:.0f})"
    else:
        bits = int(getattr(ds, "BitsStored", 16))
        lo, hi = 0.0, float(2**bits - 1)
        source = f"BitsStored({bits}bit)"
    if hi <= lo:
        lo, hi = float(dens.min()), float(dens.max()) + 1e-6
        source += " [degenerate -> fallback to dens min/max]"
    return lo, hi, source


def _air_from_gray(gray: np.ndarray):
    """YoloBmdEngine._air_reference()와 동일 로직 (테두리-연결성 + 그림자/직접노출)."""
    u8 = (np.clip(gray, 0, 1) * 255).astype(np.uint8)
    thr, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bg_mask_raw = u8 < thr
    bg_mask = YoloBmdEngine._border_connected_mask(bg_mask_raw)
    bg = gray[bg_mask]

    H, W = gray.shape
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
            return float(np.percentile(bg_fallback, 25)), _deep_frac(bg_mask_raw), "tier2b"
        return float(np.percentile(gray, 1)), 0.0, "tier3"

    bg_only = np.where(bg_mask, u8, 0).astype(np.float32)
    local_mean = cv2.blur(bg_only, (5, 5))
    local_sq_mean = cv2.blur(bg_only ** 2, (5, 5))
    local_std = np.sqrt(np.clip(local_sq_mean - local_mean ** 2, 0, None))
    direct = bg_mask & (local_std > YoloBmdEngine.AIR_SHADOW_STD_MAX)
    if direct.sum() >= 0.005 * gray.size:
        return float(np.percentile(gray[direct], 25)), _deep_frac(direct), "tier1"
    return float(np.percentile(gray[bg_mask], 25)), _deep_frac(bg_mask), "tier2"


def analyze(path: Path):
    ds, dens, mono1 = _load_dens(path)

    # ---- 현재 방식: 이미지 자체 0.5~99.5 percentile ----
    lo_p, hi_p = np.percentile(dens, 0.5), np.percentile(dens, 99.5)
    if hi_p <= lo_p:
        lo_p, hi_p = float(dens.min()), float(dens.max()) + 1e-6
    gray_pct = np.clip((dens - lo_p) / (hi_p - lo_p), 0, 1).astype(np.float32)
    air_pct, deep_pct, tier_pct = _air_from_gray(gray_pct)

    # ---- 제안 방식: 고정 스케일 ----
    lo_f, hi_f, source = _fixed_scale_bounds(ds, dens, mono1)
    gray_fix = np.clip((dens - lo_f) / (hi_f - lo_f), 0, 1).astype(np.float32)
    air_fix, deep_fix, tier_fix = _air_from_gray(gray_fix)

    return {
        "file": path.name, "air_pct": air_pct, "deep_pct": deep_pct, "tier_pct": tier_pct,
        "air_fix": air_fix, "deep_fix": deep_fix, "tier_fix": tier_fix, "source": source,
    }


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python scripts/debug_windowing_compare.py <dicom files or folders...>")
        raise SystemExit(1)

    files = collect_files(sys.argv[1:])
    if not files:
        print("No DICOM files found.")
        raise SystemExit(1)

    rows = []
    for f in files:
        try:
            rows.append(analyze(f))
        except Exception as exc:  # noqa: BLE001
            print(f"{f.name} : [error] {type(exc).__name__}: {exc}")

    print(f"{'file':40s} {'air_pct':>8s} {'deep_pct':>9s} {'air_fix':>8s} {'deep_fix':>9s}  {'source'}")
    for r in rows:
        print(
            f"{r['file']:40s} {r['air_pct']:8.4f} {r['deep_pct']:9.3f} "
            f"{r['air_fix']:8.4f} {r['deep_fix']:9.3f}  {r['source']}"
        )

    if rows:
        air_p = np.array([r["air_pct"] for r in rows])
        air_f = np.array([r["air_fix"] for r in rows])
        deep_p = np.array([r["deep_pct"] for r in rows])
        deep_f = np.array([r["deep_fix"] for r in rows])
        print()
        print(f"n={len(rows)}")
        print(f"air_pct : median={np.median(air_p):.3f}  p90={np.percentile(air_p,90):.3f}  max={air_p.max():.3f}")
        print(f"air_fix : median={np.median(air_f):.3f}  p90={np.percentile(air_f,90):.3f}  max={air_f.max():.3f}")
        print(f"deep_pct: median={np.median(deep_p):.3f}  >=0.3: {(deep_p>=0.3).sum()}/{len(rows)}")
        print(f"deep_fix: median={np.median(deep_f):.3f}  >=0.3: {(deep_f>=0.3).sum()}/{len(rows)}")
        delta_air = air_f - air_p
        print(f"delta(air_fix-air_pct): median={np.median(delta_air):+.4f}  |max|={np.abs(delta_air).max():.4f}")


if __name__ == "__main__":
    main()
