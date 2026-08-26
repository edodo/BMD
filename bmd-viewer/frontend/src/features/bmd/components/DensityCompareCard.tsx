// 종적 대조 카드 + 격자 렌더러 — CompareView(같은 환자, 시간축)와
// MultiPatientCompareView(여러 환자, 기준 1건 vs 비교 여러 건) 둘 다 이
// 컴포넌트를 그대로 재사용한다. 두 화면의 차이는 "어떤 스터디를 pre/post로
// 고르느냐"뿐, 카드 자체가 비교하는 방식(이미지+히트맵+픽셀격자+5구역격자)은
// 동일해야 하므로 여기 한 곳에서만 구현한다.
import { useMemo } from "react";
import { getDataProvider } from "@/lib/data";
import { config } from "@/lib/config";
import {
  REGION_NAMES,
  REGION_LABELS,
  REGION_GRID_POS,
  computeDensityRegionDeltas,
  countDensityCellChanges,
  pctChange,
} from "@/features/bmd/regions";
import type { XrayStudyDetail } from "@/lib/types";

const dp = getDataProvider();

export function shortDate(iso: string | null | undefined): string {
  if (!iso) return "—";
  return new Date(iso).toLocaleDateString("en-CA");
}
export function fmtPct(v: number | null): string {
  if (v == null) return "—";
  return `${v >= 0 ? "+" : ""}${v.toFixed(1)}%`;
}

// 기존엔 post−pre 정규화 밀도를 프론트에서 N×N <div> 격자로 색칠했는데,
// ROI 픽셀 해상도가 격자 칸 수보다 작을 때 빈 셀이 생기는 리샘플링
// 아티팩트(줄무늬)에 취약했다. 지금은 서버가 matplotlib으로 bilinear
// 보간 + 고정 컬러스케일 + 컬러바로 렌더링한 PNG를 그대로 보여준다
// (density-diff.png, comparisons.py) — L1~L5 BAR 히트맵과 같은 렌더링
// 관례를 그대로 따르고, 칸 경계/빈 셀 아티팩트가 없다.
function PixelDensityDiffImage({
  preStudyId,
  postStudyId,
  preGrid,
  postGrid,
}: {
  preStudyId: string | undefined;
  postStudyId: string;
  preGrid: (number | null)[][] | null;
  postGrid: (number | null)[][] | null;
}) {
  const counts = useMemo(
    () => countDensityCellChanges(preGrid, postGrid, config.comparePixelDeltaAbs),
    [preGrid, postGrid]
  );
  if (!preStudyId) return null;
  return (
    <div className="density-diff">
      <img
        className="cmp-img density-diff-img"
        src={dp.densityDiffUrl(preStudyId, postStudyId)}
        alt="Pixel-level density change vs reference"
      />
      <p className="bone-loss-cap">
        Pixel-level density change vs reference — ↓ {counts.lossCells} (blue) / ↑{" "}
        {counts.gainCells} (red) of {counts.gridCells} cells (|Δ| &gt;{" "}
        {config.comparePixelDeltaAbs})
      </p>
    </div>
  );
}

// post−pre 밀도를 5구역으로 나눠 색 + % 로 보여주는 3x3(5칸만 사용) 격자.
function DensityRegionGrid({
  pre,
  post,
  aspect,
}: {
  pre: (number | null)[][] | null;
  post: (number | null)[][] | null;
  aspect: number | null;
}) {
  const deltas = useMemo(() => computeDensityRegionDeltas(pre, post), [pre, post]);
  if (!pre || !post) return null;
  const byRegion = Object.fromEntries(deltas.map((d) => [d.region, d]));

  return (
    <div className="region-grid-wrap">
      <div className="region-grid" style={{ aspectRatio: aspect ?? 1 }}>
        {REGION_NAMES.map((region) => {
          const [row, col] = REGION_GRID_POS[region];
          const d = byRegion[region];
          const pct = d?.deltaPct ?? null;
          const loss = pct != null && pct < -config.compareRegionDeltaPct;
          const gain = pct != null && pct > config.compareRegionDeltaPct;
          const signal = loss ? "loss" : gain ? "gain" : "flat";
          return (
            <div
              key={region}
              className={`region-cell region-${signal}`}
              style={{ gridRow: row + 1, gridColumn: col + 1 }}
              title={REGION_LABELS[region]}
            >
              <span className="region-label">{REGION_LABELS[region]}</span>
              <span className="region-pct">{fmtPct(pct)}</span>
            </div>
          );
        })}
      </div>
      <p className="bone-loss-cap">
        Density change vs reference, by region — blue = decrease, red = increase
        (|Δ| &gt; {config.compareRegionDeltaPct}%)
      </p>
    </div>
  );
}

export function CompareCard({
  study,
  role,
  preGrid,
  preBmd,
  preStudyId,
  contrast,
  patientLabel,
  preTagText = "Pre-op",
  postTagText = "Post-op",
}: {
  study: XrayStudyDetail;
  role: "pre" | "post";
  preGrid: (number | null)[][] | null;
  preBmd: number | null;
  /** 픽셀 diff PNG 렌더링용 기준 스터디 id (role="post" 카드에서만 씀). */
  preStudyId?: string;
  contrast: boolean;
  /** 다중 환자 비교에서만 씀 — 어느 환자의 검사인지 카드에 표시. */
  patientLabel?: string;
  /** 같은 환자 시간축 비교("Pre-op"/"Post-op")와 여러 환자 기준 비교
   *  ("Reference"/"Compare")가 태그 문구만 다르게 쓸 수 있도록. */
  preTagText?: string;
  postTagText?: string;
}) {
  const m = study.measurement;
  const l4 = dp.assetUrl(m?.l4_crop_path ?? null);
  const dens = dp.assetUrl(m?.xai_overlay_path ?? null);
  const overallPct =
    role === "post" ? pctChange(preBmd, m?.bmd_value ?? null) : null;
  return (
    <div className={`cmp-card ${role}`}>
      {patientLabel && <div className="cmp-patient-label">{patientLabel}</div>}
      <div className="cmp-card-head">
        <span className={`cmp-tag ${role}`}>
          {role === "pre" ? preTagText : postTagText}
        </span>
        <span className="cmp-date">{shortDate(study.acquired_at)}</span>
        <span className="cmp-bmd">
          BMD {m?.bmd_value != null ? m.bmd_value.toFixed(3) : "—"}
        </span>
        {overallPct != null && (
          <span className={`cmp-delta ${overallPct >= 0 ? "up" : "down"}`}>
            {fmtPct(overallPct)}
          </span>
        )}
      </div>
      {l4 ? (
        <img
          className="cmp-img"
          src={`${l4}?v=${study.id}`}
          alt="Extracted L4"
        />
      ) : (
        <div className="cmp-img placeholder">No L4 crop</div>
      )}
      {dens && (
        <img
          className="cmp-img"
          src={`${dens}?v=${study.id}`}
          alt="Density basis"
        />
      )}
      {role === "post" && contrast && (
        <>
          <PixelDensityDiffImage
            preStudyId={preStudyId}
            postStudyId={study.id}
            preGrid={preGrid}
            postGrid={m?.l4_density_grid ?? null}
          />
          <DensityRegionGrid
            pre={preGrid}
            post={m?.l4_density_grid ?? null}
            aspect={m?.l4_density_grid_aspect ?? null}
          />
        </>
      )}
    </div>
  );
}
