// 밀도 근거 패널: L4 ROI를 실제 감쇠로 색칠한 히트맵 + 숫자 컬러바.
// "Extracted L4 vertebra" 크롭과 같은 크기로 나란히 배치하기 위해 별도 컴포넌트로 분리.
// (배경·금속 제외, 국소 연부조직 기준 → BMD 숫자의 직접 근거)

// 감쇠(attenuation) 값 표시: 정수 반올림 + 천단위 구분.
function fmtAtten(v: number | null | undefined): string {
  if (v == null || Number.isNaN(v)) return "—";
  return Math.round(v).toLocaleString();
}

// 정상/골감소 존 경계(BMD 지수 0~1). 이 proxy 지수에는 아직 검증된 임상
// 컷오프가 없으므로 기본값은 '미표시(null)'. DXA로 보정한 뒤에만
// VITE_BMD_NORMAL_THRESHOLD(예: 0.42)로 설정해 존을 켠다. (표시 전용, 재추론 불필요)
const NORMAL_THRESHOLD: number | null = (() => {
  const raw = import.meta.env.VITE_BMD_NORMAL_THRESHOLD;
  const v = raw != null ? Number(raw) : NaN;
  return Number.isFinite(v) && v > 0 && v < 1 ? v : null;
})();

export function DensityBasis({
  densityUrl,
  densityLow,
  densityHigh,
  roiMean,
  bmdValue,
}: {
  densityUrl: string;
  densityLow?: number | null;
  densityHigh?: number | null;
  roiMean?: number | null;
  bmdValue?: number | null;
}) {
  const hasScale = densityLow != null && densityHigh != null;
  const markerPct =
    bmdValue != null ? Math.min(100, Math.max(0, bmdValue * 100)) : null;

  return (
    <div className="density-basis xai-heatmap">
      <h4>Density basis — L4 ROI attenuation</h4>
      <img src={densityUrl} alt="L4 ROI density heatmap (actual attenuation)" />

      {hasScale ? (
        <div className="dens-gauge">
          {/* 상단 말풍선: 이 L4의 평균 위치와 BMD 지수 */}
          {markerPct != null && (
            <span
              className="dens-callout"
              style={{ left: `${Math.min(88, Math.max(12, markerPct))}%` }}
            >
              ★ Mean: {fmtAtten(roiMean)}
              {bmdValue != null ? ` (BMD ${bmdValue.toFixed(3)})` : ""}
            </span>
          )}
          {/* 존 가이드라인 + 그라데이션 바 (수직선이 관통) */}
          <div className="dens-track">
            {markerPct != null && (
              <span
                className="dens-needle"
                style={{ left: `${markerPct}%` }}
                aria-hidden="true"
              />
            )}
            {NORMAL_THRESHOLD != null && (
              <div className="dens-zoneline">
                <span className="z lo">Low density (osteopenia)</span>
                <span
                  className="dens-zone-div"
                  style={{ left: `${NORMAL_THRESHOLD * 100}%` }}
                  aria-hidden="true"
                />
                <span className="z hi">Normal density</span>
              </div>
            )}
            <div className="dens-bar">
              <span className="bargrad" aria-hidden="true" />
            </div>
          </div>
          {/* 최솟값 / 최댓값 (작고 낮은 대비) */}
          <div className="dens-minmax">
            <span>
              {fmtAtten(densityLow)}
              <em> Min</em>
            </span>
            <span>
              {fmtAtten(densityHigh)}
              <em> Max</em>
            </span>
          </div>
        </div>
      ) : (
        <div className="xai-scale">
          <span className="muted">Low</span>
          <span className="bargrad" aria-hidden="true" />
          <span className="muted">High density</span>
        </div>
      )}

      {hasScale ? (
        <>
          {/* 텍스트: 색상 의미만 (수치는 위 게이지에 분리 표기) */}
          <p className="dens-legend">
            Per-image scale —{" "}
            <span className="lg blue">soft tissue (blue)</span> →{" "}
            <span className="lg red">densest bone (red)</span>. <br />
            Air background & metal excluded (robust to framing / hip prostheses); <br />
            warm = denser bone.
          </p>
          {/* 주의사항: 시각적 콜아웃으로 분리 */}
          {NORMAL_THRESHOLD != null ? (
            <div className="dens-caveat warn">
              <span className="ic" aria-hidden="true">
                ⚠
              </span>
              <p>
                Normal / low split at index{" "}
                <b>{NORMAL_THRESHOLD.toFixed(2)}</b> is a configured threshold —{" "}
                <b>validate against DXA</b> before clinical use.
              </p>
            </div>
          ) : (
            <div className="dens-caveat info">
              <span className="ic" aria-hidden="true">
                ℹ
              </span>
              <p>
                <b>Proxy index — not DXA-calibrated.</b> <br />
                No normal / abnormal cutoff; <br />
                Read as <b>within-patient change</b>, not an absolute diagnosis.
              </p>
            </div>
          )}
        </>
      ) : (
        <p className="desc">
          Per-pixel attenuation inside the L4 ROI <br />
          — this is exactly what the BMD value is computed from. <br />
          Warm = denser bone (raises BMD).
        </p>
      )}
    </div>
  );
}
