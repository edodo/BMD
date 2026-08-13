// L1–L5 panel: BAR + QC for every detected vertebra, not just the target.
// Reuses the same green/amber/red traffic-light language as XaiPanel (tl-dot/tl-badge)
// so a clinician reads "trustworthy" the same way in both places.
import type { VertebraSegment } from "@/lib/types";
import { ordinalPct } from "@/features/bmd/format";
import { ClickToZoom } from "@/components/ui/ClickToZoom";

const QC_LABEL: Record<string, string> = {
  PASS: "OK",
  WARN: "Caution",
  FAIL: "Unreliable",
};

const QC_SIGNAL: Record<string, "good" | "caution" | "risk"> = {
  PASS: "good",
  WARN: "caution",
  FAIL: "risk",
};

const LEVEL_ORDER = ["L1", "L2", "L3", "L4", "L5"];

// Cohort-relative AE reconstruction-error percentile (v18 Sec.C9) -- NOT a
// diagnosis, just "how unusual this vertebra's texture is vs the reference
// cohort". Thresholds are placeholders until real outcome labels exist.
function anomalySignal(pct: number): "good" | "caution" | "risk" {
  if (pct >= 90) return "risk";
  if (pct >= 70) return "caution";
  return "good";
}

const FEATURE_LABELS: Record<string, string> = {
  bar: "Bone attenuation ratio",
  glcm_contrast: "Texture contrast",
  glcm_correlation: "Texture correlation",
  glcm_energy: "Texture energy (uniformity)",
  glcm_homogeneity: "Texture homogeneity",
  glcm_entropy: "Texture entropy (randomness)",
  vario_slope: "Variogram slope (TBS-like)",
  fractal_dim: "Trabecular fractal dimension",
};

// SHAP(KernelExplainer) attribution -> a short "driven by" tooltip, most
// influential feature first (by |contribution|). Sign tells direction:
// + pushed the reconstruction error up (more unusual), - pulled it down.
function shapTooltip(shap: Record<string, number> | null): string {
  const base =
    "Texture anomaly score: this vertebra's reconstruction error vs. the reference cohort's distribution. Cohort-relative only -- not a diagnosis.";
  if (!shap) return base;
  const top = Object.entries(shap)
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
    .slice(0, 3)
    .map(([k, v]) => `${FEATURE_LABELS[k] ?? k} (${v >= 0 ? "+" : ""}${v.toFixed(3)})`)
    .join(", ");
  return `${base}\nDriven by: ${top}`;
}

export function VertebraPanel({
  segments,
  barMapUrl,
}: {
  segments: VertebraSegment[];
  barMapUrl?: string | null;
}) {
  const byLabel = new Map(segments.map((s) => [s.label, s]));
  const rows = LEVEL_ORDER.map((label) => byLabel.get(label)).filter(
    (s): s is VertebraSegment => s != null
  );
  if (rows.length === 0) return null;

  return (
    <div className="vertebra-panel">
      <h4>L1–L5 bone attenuation ratio (BAR)</h4>
      <p className="muted">
        Every detected vertebra on this film, not just the target vertebra
        used for the headline value above.
      </p>
      {barMapUrl && (
        <div className="bar-map-l1l5">
          <ClickToZoom
            src={barMapUrl}
            alt="Per-vertebra BAR contribution heatmap, L1–L5"
          />
        </div>
      )}
      <ul className="vertebra-list">
        {rows.map((s) => {
          const signal = s.qc_status ? QC_SIGNAL[s.qc_status] ?? "info" : "info";
          return (
            <li key={s.label} className={s.is_target ? "is-target" : undefined}>
              <span className="v-label">
                {s.label}
                {s.is_target && <span className="v-target-badge">target</span>}
              </span>
              <span className="v-bar">{s.bar != null ? s.bar.toFixed(3) : "—"}</span>
              <span className={`tl-badge ${signal}`}>
                <span className={`tl-dot ${signal}`} aria-hidden="true" />
                {s.qc_status ? QC_LABEL[s.qc_status] ?? s.qc_status : "n/a"}
              </span>
              {s.qc_message && (
                <span className="v-qc-msg" title={s.qc_message} aria-label={s.qc_message}>
                  ⓘ
                </span>
              )}
              {s.anomaly_pct != null && (
                <span
                  className={`tl-badge v-anomaly ${anomalySignal(s.anomaly_pct)}`}
                  title={shapTooltip(s.anomaly_shap)}
                >
                  <span
                    className={`tl-dot ${anomalySignal(s.anomaly_pct)}`}
                    aria-hidden="true"
                  />
                  {ordinalPct(s.anomaly_pct)}
                </span>
              )}
            </li>
          );
        })}
      </ul>
    </div>
  );
}
