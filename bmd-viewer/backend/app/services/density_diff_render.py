"""두 스터디의 L4 밀도 격자(l4_density_grid) 차이를 matplotlib으로 렌더링.

기존 프론트엔드 HTML/CSS 격자(색칠된 <div> 셀)는 칸 경계가 딱딱하고, ROI
픽셀 해상도가 격자 칸 수(n)보다 작을 때 빈 셀이 생기는 리샘플링 아티팩트에
취약했다(사용자 리포트: 줄무늬). matplotlib으로 서버에서 PNG를 렌더링하면
bilinear 보간으로 매끈하게 그릴 수 있고, L1~L5 BAR 히트맵(_bar_style)과
같은 고정 컬러 스케일/컬러바 관례를 그대로 따를 수 있다.
"""
from __future__ import annotations

import io

import numpy as np

# 요청마다 min/max로 다시 스케일을 잡으면 같은 색이 비교마다 다른 값을
# 뜻하게 된다(BAR_COLOR_VMIN/VMAX와 같은 이유로 고정 -- yolo_engine.py의
# _bar_style() 주석 참고: 예전에 스터디마다 percentile로 재스케일했다가
# "기준이 자꾸 바뀐다"는 리포트로 되돌린 전례가 있다). 그렇다고 선형(Normalize)
# 고정 스케일을 쓰면, 이번 실측(BAR 0.193->0.955)처럼 변화 폭이 큰 스터디는
# 대부분의 셀이 최대색으로 그냥 클리핑돼(사용자 리포트: "전체가 한 색으로
# 포화됨") 그 안의 세부 차이가 안 보인다. SymLogNorm(대칭 로그)을 쓰면 스케일
# 자체는 여전히 고정이라 비교 가능성은 그대로 유지하면서, |Δ|<=linthresh
# 구간은 선형(작은 변화도 잘 구분), 그 밖은 로그 압축(큰 변화도 뭉개지지
# 않고 그라데이션 유지)으로 정밀하게 보여준다.
DIFF_LINTHRESH = 0.05  # 이 폭까지는 선형 -- config.comparePixelDeltaAbs와 같은 값
DIFF_COLOR_ABS_MAX = 1.5  # 이 이상은 로그 압축이라 vmax를 넉넉히 잡아도 안 뭉갠다


def _cell(grid: list[list[float | None]], n_ref: int, i: int, j: int) -> float | None:
    """grid가 n_ref×n_ref가 아니어도(예: 서로 다른 시점에 다른 격자 해상도로
    처리된 스터디끼리 비교) 상대 위치(bbox 기준 정규화 위치)로 값을 찾는다."""
    n = len(grid)
    if n == 0:
        return None
    gi = min(n - 1, int(i * n / n_ref))
    row = grid[gi]
    gj = min(len(row) - 1, int(j * n / n_ref)) if row else -1
    return row[gj] if 0 <= gj < len(row) else None


def render_density_diff_png(
    pre_grid: list[list[float | None]],
    post_grid: list[list[float | None]],
    aspect: float | None,
    pre_bar: float | None = None,
    post_bar: float | None = None,
) -> bytes:
    """post − pre 격자 차이를 diverging 컬러맵 PNG로 렌더링해 바이트로 반환.

    pre_bar/post_bar(각 스터디의 헤드라인 BAR 값)는 색으로만 표시되던 '기준
    (reference)'이 실제로 어떤 수치였는지를 제목에 그대로 적어 보여준다
    (사용자 리포트: "지금은 뭐가 기준인지 모르겠네" -- 변화 폭이 큰 스터디는
    거의 전체가 한 색(빨강)에 가까워, 색만 봐서는 기준값이 뭐였는지 알 길이
    없었다).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm

    n = len(post_grid)
    diff = np.full((n, n), np.nan, dtype=np.float32)
    for i in range(n):
        row = post_grid[i]
        for j in range(len(row)):
            q = row[j]
            p = _cell(pre_grid, n, i, j)
            if p is not None and q is not None:
                diff[i, j] = q - p

    masked = np.ma.masked_invalid(diff)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="white", alpha=0.0)
    norm = SymLogNorm(
        linthresh=DIFF_LINTHRESH,
        vmin=-DIFF_COLOR_ABS_MAX,
        vmax=DIFF_COLOR_ABS_MAX,
    )

    fig_h = 4.2
    fig_w = max(2.5, fig_h * (aspect or 1.0))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(
        masked, cmap=cmap, norm=norm, interpolation="bilinear", aspect="auto"
    )
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cbar.set_label("Density change (BAR scale, log-compressed beyond ±0.05)", fontsize=8)
    cbar.ax.tick_params(labelsize=8)
    if pre_bar is not None and post_bar is not None:
        title = f"Density change — reference BAR {pre_bar:.3f} → this study BAR {post_bar:.3f}"
    else:
        title = "Pixel-level density change vs reference"
    ax.set_title(title, fontsize=10)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()
