// Cohort Bone Health Index (BHI) chip for the target vertebra (v18_spec.md Sec.5/D0).
// Composite z-score (BAR 50% + texture 50%, weights/signs learned from this deployment's
// own reference cohort, never assumed) reduced to a 0-100 cohort percentile + a WHO-style
// category label applied to the BAR z-score alone. Deliberately NOT called "T-score" or
// rendered next to one -- this is cohort-relative, not DXA-calibrated (v18 Sec.4 honesty
// principle: "T-score 아님 -- 코호트 z-score").
import type { VertebraSegment } from "@/lib/types";
import { ordinalPct } from "@/features/bmd/format";

function bhiSignal(category: string | null): "good" | "caution" | "risk" {
  if (!category) return "good";
  if (category.startsWith("Very low")) return "risk";
  if (category.startsWith("Low")) return "caution";
  return "good";
}

export function BhiChip({ segments }: { segments: VertebraSegment[] }) {
  const target = segments.find((s) => s.is_target);
  if (!target || target.bhi_pct == null) return null;

  const signal = bhiSignal(target.category);
  return (
    <div
      className={`bhi-chip ${signal}`}
      title="Cohort-relative composite z-score (BAR-weighted + trabecular texture) against this deployment's own reference cohort. NOT a DXA T-score and not a diagnosis -- a label-free proxy index only."
    >
      <span className="bhi-label">Cohort BHI</span>
      <span className={`tl-badge ${signal}`}>
        <span className={`tl-dot ${signal}`} aria-hidden="true" />
        {ordinalPct(target.bhi_pct)}
      </span>
      {target.category && <span className="bhi-category">{target.category}</span>}
    </div>
  );
}
