// 5구역 종적 대조(중앙+네 모서리) 공유 유틸.
// 밀도(기존 NxN 격자를 프론트에서 재집계)와 텍스처(백엔드가 이미 5구역으로
// 계산해 보냄) 양쪽 팝업이 "같은 중앙"을 뜻하도록, 분할 규칙은 백엔드
// yolo_engine.py의 _five_region_masks()와 정확히 동일해야 한다:
// bbox 기준 정규화 위치 (fx,fy) in [0,1]에서 중앙 50%x50%는 "center",
// 나머지는 fx<0.5/fy<0.5로 결정되는 사분면.
import type { CompareRegionName, TextureRegionFeatures } from "@/lib/types";

export const REGION_NAMES: CompareRegionName[] = [
  "top_left",
  "top_right",
  "center",
  "bottom_left",
  "bottom_right",
];

export const REGION_LABELS: Record<CompareRegionName, string> = {
  center: "Center",
  top_left: "Top-left",
  top_right: "Top-right",
  bottom_left: "Bottom-left",
  bottom_right: "Bottom-right",
};

// 3x3 CSS grid 배치 (row, col), 0-indexed — 나머지 4칸은 비워둔다.
export const REGION_GRID_POS: Record<CompareRegionName, [number, number]> = {
  top_left: [0, 0],
  top_right: [0, 2],
  center: [1, 1],
  bottom_left: [2, 0],
  bottom_right: [2, 2],
};

function regionOf(fx: number, fy: number): CompareRegionName {
  const isCenter = fx >= 0.25 && fx <= 0.75 && fy >= 0.25 && fy <= 0.75;
  if (isCenter) return "center";
  if (fy < 0.5) return fx < 0.5 ? "top_left" : "top_right";
  return fx < 0.5 ? "bottom_left" : "bottom_right";
}

export interface DensityRegionDelta {
  region: CompareRegionName;
  avgPre: number | null;
  avgPost: number | null;
  deltaPct: number | null; // (avgPost-avgPre)/avgPre*100
  cellCount: number;
}

/** NxN 밀도 격자(post/pre) -> 5구역별 평균 델타/퍼센트. 백엔드 _density_grid()가
 * bbox를 linspace(0..n)로 균등 분할하므로, 셀 (i,j)의 정규화 중심은
 * ((j+0.5)/n, (i+0.5)/n) — _five_region_masks()의 픽셀 단위 fx/fy와 같은 뜻. */
export function computeDensityRegionDeltas(
  pre: (number | null)[][] | null,
  post: (number | null)[][] | null
): DensityRegionDelta[] {
  const empty = REGION_NAMES.map((region) => ({
    region,
    avgPre: null,
    avgPost: null,
    deltaPct: null,
    cellCount: 0,
  }));
  if (!pre || !post || post.length === 0) return empty;
  const n = post.length;
  const sums: Record<CompareRegionName, { pre: number; post: number; n: number }> = {
    center: { pre: 0, post: 0, n: 0 },
    top_left: { pre: 0, post: 0, n: 0 },
    top_right: { pre: 0, post: 0, n: 0 },
    bottom_left: { pre: 0, post: 0, n: 0 },
    bottom_right: { pre: 0, post: 0, n: 0 },
  };
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const p = pre[i]?.[j];
      const q = post[i]?.[j];
      if (p == null || q == null) continue;
      const fx = (j + 0.5) / n;
      const fy = (i + 0.5) / n;
      const r = regionOf(fx, fy);
      sums[r].pre += p;
      sums[r].post += q;
      sums[r].n += 1;
    }
  }
  return REGION_NAMES.map((region) => {
    const s = sums[region];
    if (s.n === 0) {
      return { region, avgPre: null, avgPost: null, deltaPct: null, cellCount: 0 };
    }
    const avgPre = s.pre / s.n;
    const avgPost = s.post / s.n;
    const deltaPct = avgPre !== 0 ? ((avgPost - avgPre) / avgPre) * 100 : null;
    return { region, avgPre, avgPost, deltaPct, cellCount: s.n };
  });
}

export interface DensityCellChangeCounts {
  lossCells: number;
  gainCells: number;
  gridCells: number; // n*n (전체 격자 칸 수 — 유효 셀만이 아니라 old UI와 동일하게 분모로 씀)
}

/** NxN 밀도 격자(post/pre)에서 |Δ|가 thresholdAbs를 넘는 셀 수를 센다.
 * density-diff.png(matplotlib 렌더링)는 이미지라 셀 개수를 직접 셀 수 없으므로,
 * 프론트가 이미 갖고 있는 원본 격자로 별도 계산해 이미지 아래 캡션에 쓴다. */
export function countDensityCellChanges(
  pre: (number | null)[][] | null,
  post: (number | null)[][] | null,
  thresholdAbs: number
): DensityCellChangeCounts {
  if (!pre || !post) return { lossCells: 0, gainCells: 0, gridCells: 0 };
  const n = post.length;
  let lossCells = 0;
  let gainCells = 0;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const p = pre[i]?.[j];
      const q = post[i]?.[j];
      if (p == null || q == null) continue;
      const d = q - p;
      if (d < -thresholdAbs) lossCells += 1;
      else if (d > thresholdAbs) gainCells += 1;
    }
  }
  return { lossCells, gainCells, gridCells: n * n };
}

/** 두 값 사이의 % 변화. pre가 0/null이면 계산 불가(null). */
export function pctChange(pre: number | null, post: number | null): number | null {
  if (pre == null || post == null || pre === 0) return null;
  return ((post - pre) / pre) * 100;
}

export const TEXTURE_FEATURE_LABELS: Record<keyof TextureRegionFeatures, string> = {
  glcm_homogeneity: "Homogeneity",
  glcm_entropy: "Entropy",
  vario_slope: "Variogram slope",
  fractal_dim: "Fractal dim.",
};

export const TEXTURE_FEATURE_KEYS = Object.keys(
  TEXTURE_FEATURE_LABELS
) as (keyof TextureRegionFeatures)[];
