// XAI 패널: BMD 값에 영향을 준 항목(factors) + 모델 주목도(Eigen-CAM).
// 통계 지표를 임상 표현 + 신호등(Green-Yellow-Red) 게이지로 치환해, 의료진이
// 각 지표의 신뢰도/위험도를 한눈에 파악하도록 한다.
// 밀도 근거 히트맵은 DensityBasis 컴포넌트로 분리되어 L4 크롭 옆에 배치된다.
import type { XaiFactor } from "@/lib/types";
import { signalOrder, toClinical, type Signal } from "@/features/bmd/xaiClinical";

const SIGNAL_LABEL: Record<Signal, string> = {
  good: "Normal",
  caution: "Caution",
  risk: "Risk",
  info: "Info",
};

function TrafficGauge({ signal }: { signal: Signal }) {
  return (
    <div className={`tl-gauge sig-${signal}`} aria-hidden="true">
      <span className="seg good" />
      <span className="seg caution" />
      <span className="seg risk" />
    </div>
  );
}

export function XaiPanel({
  factors,
  camUrl,
}: {
  factors: XaiFactor[];
  camUrl?: string | null;
}) {
  // 임상 표현으로 변환 후 위험도 순으로 정렬(위험한 지표부터).
  const items = factors
    .map((f) => ({ f, c: toClinical(f) }))
    .sort((a, b) => signalOrder[a.c.signal] - signalOrder[b.c.signal]);

  return (
    <div className="xai-panel">
      <h4>Basis for Judgment (XAI)</h4>
      <p className="muted">
        Signal quality per factor (heuristic reliability guide)
      </p>
      <ul className="xai-clinical">
        {items.map(({ f, c }) => (
          <li key={f.label}>
            <div className="row">
              <span className="lbl">
                <span className={`tl-dot ${c.signal}`} aria-hidden="true" />
                {c.clinical}
              </span>
              <span className={`tl-badge ${c.signal}`}>
                {SIGNAL_LABEL[c.signal]}
              </span>
            </div>
            <TrafficGauge signal={c.signal} />
            <p className="desc">
              {c.meaning}
              {f.description ? ` · ${f.description}` : ""}
            </p>
          </li>
        ))}
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
