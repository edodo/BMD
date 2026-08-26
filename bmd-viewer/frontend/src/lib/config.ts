// 소스코드 UI/임상 설정값 (자주 바뀌지 않으므로 코드로 관리).
export const config = {
  // X-ray History: 페이지당 항목 수
  historyPageSize: 5,

  // 임상 경고 폴백값: 직전 검사 대비 BMD가 이 % 이상 급락하면
  // "Significant Bone Loss" 경고를 표시 (단기 급격한 골손실).
  // /precision/calibrate로 실측 LSC%가 산출되면 이 값 대신 그것을 쓴다
  // (미검증 임의값 대신 실제 재위치 노이즈 기준 사용).
  significantLossPct: 10,

  // 종적 대조(compare) 5구역 격자: post−pre 밀도 변화가 이 값(%포인트)보다
  // 더 크면 증가(붉은색), 더 작으면(음수) 감소(파란색)로 표시.
  compareRegionDeltaPct: 5,

  // 픽셀 diff 이미지(density-diff.png) 밑에 표시하는 "↓N/↑M cells" 요약의
  // 판정 기준. 이미지 자체는 서버가 렌더링하므로, 이 카운트는 프론트가 이미
  // 갖고 있는 원본 NxN 격자로 별도 계산한다(원시 밀도 단위, 0..1 스케일).
  comparePixelDeltaAbs: 0.05,
};
