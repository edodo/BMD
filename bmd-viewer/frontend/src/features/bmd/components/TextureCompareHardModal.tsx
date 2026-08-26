// Texture comparison — "detailed" version: 5-region (center + 4 quadrants)
// texture breakdown, computed backend-side at inference time
// (measurement.l4_texture_regions) because GLCM/fractal/variogram need real
// pixel data, not just the stored scalar features. Regions are intentionally
// large (not a fine grid) -- these features are noisy on small samples.
import { Modal } from "@/components/ui/Modal";
import {
  REGION_NAMES,
  REGION_LABELS,
  REGION_GRID_POS,
  TEXTURE_FEATURE_KEYS,
  TEXTURE_FEATURE_LABELS,
  pctChange,
} from "@/features/bmd/regions";
import type { XrayStudyDetail, TextureRegions } from "@/lib/types";

function fmtPct(v: number | null): string {
  if (v == null) return "—";
  return `${v >= 0 ? "+" : ""}${v.toFixed(0)}%`;
}

function RegionPanel({
  preRegions,
  postRegions,
}: {
  preRegions: TextureRegions | null;
  postRegions: TextureRegions | null;
}) {
  if (!postRegions) {
    return <p className="muted">No detailed texture data for this study yet.</p>;
  }
  return (
    <div className="region-grid-wrap">
      <div className="region-grid texcmp-hard-grid">
        {REGION_NAMES.map((region) => {
          const [row, col] = REGION_GRID_POS[region];
          const pre = preRegions?.[region] ?? null;
          const post = postRegions[region] ?? null;
          return (
            <div
              key={region}
              className="region-cell texcmp-hard-cell"
              style={{ gridRow: row + 1, gridColumn: col + 1 }}
            >
              <span className="region-label">{REGION_LABELS[region]}</span>
              {!post ? (
                <span className="texcmp-insufficient">insufficient data</span>
              ) : (
                <ul className="texcmp-hard-features">
                  {TEXTURE_FEATURE_KEYS.map((key) => {
                    const pct = pctChange(pre?.[key] ?? null, post[key] ?? null);
                    return (
                      <li key={key}>
                        <span className="texcmp-hard-flabel">
                          {TEXTURE_FEATURE_LABELS[key]}
                        </span>
                        <span
                          className={`texcmp-delta ${
                            pct == null ? "" : pct >= 0 ? "up" : "down"
                          }`}
                        >
                          {fmtPct(pct)}
                        </span>
                      </li>
                    );
                  })}
                </ul>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

export function TextureCompareHardModal({
  pre,
  posts,
  onClose,
}: {
  pre: XrayStudyDetail;
  posts: XrayStudyDetail[];
  onClose: () => void;
}) {
  const preRegions = pre.measurement?.l4_texture_regions ?? null;
  return (
    <Modal title="Texture comparison — detailed" onClose={onClose} size="xwide">
      <p className="muted texcmp-note">
        Per-region texture change vs. pre-op (center + 4 quadrants — large regions,
        because GLCM/fractal/variogram need real pixel samples to be meaningful).
      </p>
      <div className="texcmp-hard-panels">
        {posts.map((p) => (
          <div key={p.id} className="texcmp-hard-panel">
            <h4>{new Date(p.acquired_at ?? p.uploaded_at).toLocaleDateString("en-CA")}</h4>
            <RegionPanel
              preRegions={preRegions}
              postRegions={p.measurement?.l4_texture_regions ?? null}
            />
          </div>
        ))}
      </div>
    </Modal>
  );
}
