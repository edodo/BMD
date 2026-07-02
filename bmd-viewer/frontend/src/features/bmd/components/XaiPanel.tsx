// XAI 패널: BMD 값에 영향을 준 항목을 의사가 이해할 수 있게 표시.
// 항목 값 아래에는 두 가지 시각적 근거(둘 다 L4로 크롭)를 함께 보여준다:
//   1) densityUrl — L4 ROI를 실제 감쇠로 색칠 (BMD 숫자의 직접 근거)
//   2) camUrl     — L4 크롭 Eigen-CAM (분할 모델이 주목한 영역)
import type { XaiFactor } from "@/lib/types";

export function XaiPanel({
  factors,
  densityUrl,
  camUrl,
}: {
  factors: XaiFactor[];
  densityUrl?: string | null;
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

      {densityUrl && (
        <div className="xai-heatmap">
          <h5>Density basis — L4 ROI attenuation</h5>
          <img src={densityUrl} alt="L4 ROI density heatmap (actual attenuation)" />
          <div className="xai-scale">
            <span className="muted">Low</span>
            <span className="bargrad" aria-hidden="true" />
            <span className="muted">High density</span>
          </div>
          <p className="desc">
            Per-pixel attenuation inside the L4 ROI — this is exactly what the
            BMD value is computed from. Warm = denser bone (raises BMD).
          </p>
        </div>
      )}

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
