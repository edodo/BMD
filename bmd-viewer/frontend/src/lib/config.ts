// 소스코드 UI/임상 설정값 (자주 바뀌지 않으므로 코드로 관리).
export const config = {
  // X-ray History: 페이지당 항목 수
  historyPageSize: 5,

  // 임상 경고: 직전 검사 대비 BMD가 이 % 이상 급락하면
  // "Significant Bone Loss" 경고를 표시 (단기 급격한 골손실).
  significantLossPct: 10,

  // 종적 대조(compare): post−pre 정규화 밀도 차이가 이 값보다 더 음수인
  // 격자 셀을 '골손실 부위(붉은 격자)'로 표시.
  compareBoneLossDelta: 0.05,
};
