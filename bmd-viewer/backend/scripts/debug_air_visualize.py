"""`_air_reference()`가 실제로 어느 영역을 가리키는지 파일별로 시각화한다.

`debug_bar_components.py`/`debug_bar_air_vs_disc.py`가 찍는 air 숫자가 진짜
무피폭 배경을 가리키는지, 아니면 콜리메이션 그림자나 다른 걸 잡고 있는지 눈으로
확인하기 위한 도구. YOLO 추론이 필요 없다(air_reference는 DICOM 픽셀만 보고
계산하므로 L4 검출과 무관) — 모델 로드 없이 DICOM 로드 + Otsu만 돌아 훨씬
가볍고 빠르다.

`_air_reference()`가 반환하는 숫자 자체는 실제 프로덕션 함수를 그대로 호출해서
얻는다(항상 실제 값과 일치). 그 숫자가 어느 픽셀들에서 나왔는지 보여주는
마스크는 같은 함수가 마스크를 반환하지 않으므로 시각화 전용으로 내부 로직을
그대로 복사해 재구성했다 — `YoloBmdEngine._air_reference`가 바뀌면 아래
`_air_mask_detail`도 같이 고쳐야 어긋나지 않는다. 둘의 값이 다르면(복사본이
낡았다는 뜻) 경고를 찍는다.

색상: 초록 = 실제 25th percentile 계산에 쓰인 "직접노출" 픽셀(대부분의 경우
air 값이 여기서 나옴). 파랑 = Otsu가 배경으로는 갈랐지만 콜리메이션 그림자로
판정돼 (직접노출 표본이 충분할 때는) 계산에서 제외된 픽셀. 직접노출 표본이
부족하면 파랑 영역 전체의 25th percentile로 폴백하고, 배경 자체가 거의 없으면
(그림에 아무 하이라이트도 없이) 전체 이미지 1st percentile로 폴백한다 — 타이틀에
어느 경로였는지 표시.

출력: 파일마다 PNG 한 장, 파일명 맨 앞에 air 값을 붙인다(예: 0.1805_xxxx.png)
— 탐색기에서 이름순 정렬만 해도 낮은/높은 air 순서로 훑어볼 수 있다.

사용법 (backend 디렉터리, pydicom/opencv/matplotlib 있는 환경 — 모델 불필요)
------------------------------------------------------------------------
    python scripts/debug_air_visualize.py <dicom 파일 또는 폴더...> [--out DIR] [--limit N]
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

from app.services.inference.yolo_engine import YoloBmdEngine

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


def _air_mask_detail(gray: np.ndarray):
    """`YoloBmdEngine._air_reference`의 로직을 그대로 복사 + 마스크/타이어 반환
    (테두리-연결성 필터링 포함, `_border_connected_mask`와 동일).

    Returns (value, bg_mask, direct_exposure_mask, tier)."""
    u8 = (np.clip(gray, 0, 1) * 255).astype(np.uint8)
    thr, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bg_mask_raw = u8 < thr
    bg_mask = YoloBmdEngine._border_connected_mask(bg_mask_raw)
    bg = gray[bg_mask]
    empty = np.zeros_like(bg_mask)
    if bg.size < 0.005 * gray.size:
        bg_fallback = gray[bg_mask_raw]
        if bg_fallback.size >= 0.005 * gray.size:
            return (
                float(np.percentile(bg_fallback, 25)), bg_mask_raw, empty,
                "tier2b: pre-filter background p25 (all border-connected candidates too small)",
            )
        return float(np.percentile(gray, 1)), empty, empty, "tier3: whole-image p1 (no real background)"

    bg_only = np.where(bg_mask, u8, 0).astype(np.float32)
    local_mean = cv2.blur(bg_only, (5, 5))
    local_sq_mean = cv2.blur(bg_only ** 2, (5, 5))
    local_std = np.sqrt(np.clip(local_sq_mean - local_mean ** 2, 0, None))
    direct_exposure = bg_mask & (local_std > YoloBmdEngine.AIR_SHADOW_STD_MAX)
    if direct_exposure.sum() >= 0.005 * gray.size:
        value = float(np.percentile(gray[direct_exposure], 25))
        return value, bg_mask, direct_exposure, "tier1: direct-exposure p25"
    return float(np.percentile(bg, 25)), bg_mask, empty, "tier2: all-background p25 (shadow-dominated)"


def visualize_one(path: Path, out_dir: Path):
    import matplotlib.pyplot as plt

    rgb, dens, H, W, *_ = YoloBmdEngine._load_dicom(path)
    gray = YoloBmdEngine._gray_lin(dens)

    air_official = YoloBmdEngine._air_reference(gray)
    air_recomputed, bg_mask, direct_exposure, tier = _air_mask_detail(gray)
    if abs(air_official - air_recomputed) > 1e-6:
        print(
            f"[warn] {path.name}: visualization copy diverged from _air_reference "
            f"({air_recomputed:.4f} vs official {air_official:.4f}) -- update _air_mask_detail"
        )

    overlay = cv2.cvtColor((np.clip(gray, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    shadow_only = bg_mask & ~direct_exposure
    overlay[shadow_only] = (0.5 * overlay[shadow_only] + 0.5 * np.array([60, 90, 255])).astype(np.uint8)
    overlay[direct_exposure] = (
        0.5 * overlay[direct_exposure] + 0.5 * np.array([60, 230, 90])
    ).astype(np.uint8)

    fig, ax = plt.subplots(figsize=(max(4.0, W / H * 5.0), 5.0))
    ax.imshow(overlay)
    ax.axis("off")
    ax.set_title(
        f"{path.name}\nair={air_official:.4f}  |  {tier}\n"
        "green = used for air percentile   blue = background but excluded (shadow)",
        fontsize=9,
    )
    fig.tight_layout()
    out_path = out_dir / f"{air_official:.4f}_{path.stem}.png"
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return air_official, tier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--out", default="figures/air_viz")
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

    tier_counts: dict[str, int] = {}
    for i, f in enumerate(files, 1):
        try:
            _air, tier = visualize_one(f, out_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] {f.name}: {type(exc).__name__}: {exc}")
            continue
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
        if i % 20 == 0 or i == len(files):
            print(f"  ...{i}/{len(files)}")

    print()
    print(f"저장 위치: {out_dir}")
    print("타이어(air 산출 경로)별 집계:")
    for tier, n in sorted(tier_counts.items()):
        print(f"  {tier}: {n}")


if __name__ == "__main__":
    main()
