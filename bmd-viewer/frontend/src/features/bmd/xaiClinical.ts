// XAI 통계 지표 → 임상 표현 + 신호등(Green-Yellow-Red) 변환.
// 목적: 의료진이 통계 용어 대신 임상적 의미로, 그리고 각 지표의 신뢰도(품질)를
// 1초 만에 색으로 파악하도록 한다. 임계값은 휴리스틱(미보정)이며 파이프라인
// 신뢰성 가늠용 가이드이지 진단이 아니다.
import type { XaiFactor } from "@/lib/types";

export type Signal = "good" | "caution" | "risk" | "info";

export interface XaiClinical {
  clinical: string; // 의료진용 라벨
  meaning: string; // 한 줄 임상적 의미
  signal: Signal; // 신호등 상태
}

// |값|이 goodMax 이하=good, cautionMax 이하=caution, 초과=risk
const band = (v: number, goodMax: number, cautionMax: number): Signal =>
  v <= goodMax ? "good" : v <= cautionMax ? "caution" : "risk";

// 신호등 정렬 우선순위(위험한 것부터)
export const signalOrder: Record<Signal, number> = {
  risk: 0,
  caution: 1,
  good: 2,
  info: 3,
};

export function toClinical(f: XaiFactor): XaiClinical {
  const c = f.contribution;
  switch (f.label) {
    case "L4 detection confidence":
      return {
        clinical: "Segmentation reliability",
        meaning: "How confidently the model located L4",
        signal: c >= 0.85 ? "good" : c >= 0.6 ? "caution" : "risk",
      };
    case "Density uniformity":
      // contribution = −변동계수(CV): 0에 가까울수록 균일(좋음)
      return {
        clinical: "Density uniformity",
        meaning: "Even trabecular texture (artifact / heterogeneity check)",
        signal: band(Math.abs(c), 0.15, 0.35),
      };
    case "Intensity dynamic range":
      return {
        clinical: "Exposure dynamic range",
        meaning: "Contrast spread — a very wide range can signal exposure issues",
        signal: band(Math.abs(c), 0.25, 0.5),
      };
    case "Distribution skew":
      return {
        clinical: "Density distribution symmetry",
        meaning: "Skew of the density histogram (0 = symmetric)",
        signal: band(Math.abs(c), 0.15, 0.35),
      };
    case "L4 ROI mean attenuation":
      return {
        clinical: "Relative bone density (L4)",
        meaning:
          "Trabecular density on the soft-tissue→cortical scale (uncalibrated)",
        signal: "info", // 측정값 자체 — 신뢰성 판정 대상 아님
      };
    default:
      return {
        clinical: f.label,
        meaning: f.description ?? "",
        signal: "caution",
      };
  }
}
