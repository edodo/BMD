// 기준값 근거 패널: BAR의 두 기준값(air, 연부조직)이 실제로 어느 픽셀에서
// 왔는지 전체 이미지 위에 색으로 표시하고, 계산식을 그대로 보여준다.
// 색상/계산식이 이미지 자체에 그려져 있으므로(백엔드
// YoloBmdEngine._build_air_soft_heatmap) 별도 DOM 게이지는 두지 않는다 —
// Density basis 패널과 같은 원칙.
import { ClickToZoom } from "@/components/ui/ClickToZoom";

export function AirSoftBasis({ imageUrl }: { imageUrl: string }) {
  return (
    <div className="air-soft-basis xai-heatmap">
      <h4>Reference basis — where air/soft-tissue values came from</h4>
      <ClickToZoom
        src={imageUrl}
        alt="Air and soft-tissue reference regions used for this BAR, with the calculation shown"
      />
      <p className="desc">
        Yellow = background pixels actually used as the air reference. Navy =
        background excluded as collimator shadow (no photons reached it).
        Green = the soft-tissue band used; sky-blue = the other side,
        rejected as likely contaminated. Red fill = L4 (target vertebra).
        <br />
        The formula below the image is the exact calculation for this study.
      </p>
    </div>
  );
}
