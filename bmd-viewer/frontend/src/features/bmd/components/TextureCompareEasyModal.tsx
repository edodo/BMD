// Texture comparison — "simple" version: scalar feature values, pre vs each
// post, as a table with % change. No new backend call — everything needed
// (glcm_homogeneity/entropy, vario_slope, fractal_dim, bhi_pct) already comes
// back on the study's own measurement.segments (target vertebra only).
import { Modal } from "@/components/ui/Modal";
import { TEXTURE_FEATURE_KEYS, TEXTURE_FEATURE_LABELS, pctChange } from "@/features/bmd/regions";
import type { XrayStudyDetail, VertebraSegment } from "@/lib/types";

function targetSegment(study: XrayStudyDetail): VertebraSegment | undefined {
  return study.measurement?.segments.find((s) => s.is_target);
}

function fmtPct(v: number | null): string {
  if (v == null) return "—";
  return `${v >= 0 ? "+" : ""}${v.toFixed(1)}%`;
}

function fmtVal(v: number | null): string {
  return v == null ? "—" : v.toFixed(3);
}

export function TextureCompareEasyModal({
  pre,
  posts,
  onClose,
}: {
  pre: XrayStudyDetail;
  posts: XrayStudyDetail[];
  onClose: () => void;
}) {
  const preSeg = targetSegment(pre);

  return (
    <Modal title="Texture comparison — simple" onClose={onClose} size="wide">
      <p className="muted texcmp-note">
        Whole-ROI texture values for the target vertebra, pre-op vs. each post-op study.
      </p>
      <div className="texcmp-easy-scroll">
        <table className="texcmp-table">
          <thead>
            <tr>
              <th>Feature</th>
              <th>Pre-op</th>
              {posts.map((p) => (
                <th key={p.id}>{new Date(p.acquired_at ?? p.uploaded_at).toLocaleDateString("en-CA")}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Cohort BHI</td>
              <td>{preSeg?.bhi_pct != null ? `${preSeg.bhi_pct}th` : "—"}</td>
              {posts.map((p) => {
                const seg = targetSegment(p);
                const delta =
                  seg?.bhi_pct != null && preSeg?.bhi_pct != null
                    ? seg.bhi_pct - preSeg.bhi_pct
                    : null;
                return (
                  <td key={p.id}>
                    {seg?.bhi_pct != null ? `${seg.bhi_pct}th` : "—"}
                    {delta != null && (
                      <span className={`texcmp-delta ${delta >= 0 ? "up" : "down"}`}>
                        {" "}({delta >= 0 ? "+" : ""}{delta.toFixed(0)})
                      </span>
                    )}
                  </td>
                );
              })}
            </tr>
            {TEXTURE_FEATURE_KEYS.map((key) => (
              <tr key={key}>
                <td>{TEXTURE_FEATURE_LABELS[key]}</td>
                <td>{fmtVal(preSeg?.[key] ?? null)}</td>
                {posts.map((p) => {
                  const seg = targetSegment(p);
                  const val = seg?.[key] ?? null;
                  const pct = pctChange(preSeg?.[key] ?? null, val);
                  return (
                    <td key={p.id}>
                      {fmtVal(val)}
                      {pct != null && (
                        <span className={`texcmp-delta ${pct >= 0 ? "up" : "down"}`}>
                          {" "}({fmtPct(pct)})
                        </span>
                      )}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Modal>
  );
}
