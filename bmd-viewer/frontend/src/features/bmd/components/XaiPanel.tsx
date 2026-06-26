// XAI 패널: BMD 값에 영향을 준 항목을 의사가 이해할 수 있게 표시.
import type { XaiFactor } from "@/lib/types";

export function XaiPanel({ factors }: { factors: XaiFactor[] }) {
  const sorted = [...factors].sort(
    (a, b) => Math.abs(b.contribution) - Math.abs(a.contribution)
  );
  const max = Math.max(...sorted.map((f) => Math.abs(f.contribution)), 0.01);

  return (
    <div className="xai-panel">
      <h4>Basis for Judgment (XAI)</h4>
      <p className="muted">Factors affecting the bone density value</p>
      <ul>
        {sorted.map((f) => {
          const pct = (Math.abs(f.contribution) / max) * 100;
          const positive = f.contribution >= 0;
          return (
            <li key={f.label}>
              <div className="row">
                <span className="lbl">{f.label}</span>
                <span className={`val ${positive ? "pos" : "neg"}`}>
                  {positive ? "+" : ""}
                  {f.contribution.toFixed(2)}
                </span>
              </div>
              <div className="bar">
                <div
                  className={`fill ${positive ? "pos" : "neg"}`}
                  style={{ width: `${pct}%` }}
                />
              </div>
              {f.description && <p className="desc">{f.description}</p>}
            </li>
          );
        })}
      </ul>
    </div>
  );
}
