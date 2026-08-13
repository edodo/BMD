// 밀도 근거 패널: L4 ROI를 BAR(배경 대비 상대 감쇠)로 색칠한 히트맵.
// 컬러바/스케일이 이미지 자체에 그려져 있으므로(0=연부조직 중심 diverging/
// sequential 컬러맵 + 흰 트라베큘러 ROI 윤곽선, 노트북 v18 E2와 동일 스타일)
// 별도의 DOM 게이지는 두지 않는다 — 원시 감쇠 스케일과 BAR 스케일이 따로
// 놀면(과거 v17 초기 버그의 원인) 오해를 부른다.
// "Extracted L4 vertebra" 크롭과 나란히 배치하기 위해 별도 컴포넌트로 분리.
import { ClickToZoom } from "@/components/ui/ClickToZoom";

export function DensityBasis({
  densityUrl,
  bmdValue,
}: {
  densityUrl: string;
  bmdValue?: number | null;
}) {
  return (
    <div className="density-basis xai-heatmap">
      <h4>Density basis — L4 ROI attenuation</h4>
      <ClickToZoom
        src={densityUrl}
        alt="L4 ROI BAR heatmap (per-pixel attenuation, 0 = soft tissue)"
      />
      <p className="desc">
        Per-pixel BAR (Bone Attenuation Ratio) inside the L4 trabecular ROI
        (white outline) — this is exactly what the BAR value is computed
        from.{" "}
        {bmdValue != null && (
          <>
            Current BAR: <b>{bmdValue.toFixed(3)}</b>.
          </>
        )}
        <br />
        Colour is centred at 0 = soft tissue: warm = denser bone, blue = below
        soft-tissue level. Proxy index — not DXA-calibrated.
      </p>
    </div>
  );
}
