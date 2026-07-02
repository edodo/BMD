// XAI 패널: BMD 값에 영향을 준 항목(factors) + 모델 주목도(Eigen-CAM).
// 밀도 근거 히트맵은 DensityBasis 컴포넌트로 분리되어 L4 크롭 옆에 배치된다.
import type { XaiFactor } from "@/lib/types";

export function XaiPanel({
  factors,
  camUrl,
}: {
  factors: XaiFactor[];
  camUrl?: string | null;
}) {
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

      {camUrl && (
        <div className="xai-heatmap">
          <h5>Model attention — L4 Eigen-CAM</h5>
          <img src={camUrl} alt="L4-cropped Eigen-CAM attribution" />
          <div className="xai-scale">
            <span className="muted">Low</span>
            <span className="bargrad" aria-hidden="true" />
            <span className="muted">High attention</span>
          </div>
          <p className="desc">
            Where the segmentation model focused within the L4 crop
            (class-agnostic Eigen-CAM) — context, not the density calculation.
          </p>
        </div>
      )}
    </div>
  );
}
