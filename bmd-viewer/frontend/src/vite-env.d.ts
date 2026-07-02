/// <reference types="vite/client" />

interface ImportMetaEnv {
  // 정상/골감소 존 경계 BMD 지수(예: "0.42"). 미설정 시 존 경계 미표시.
  readonly VITE_BMD_NORMAL_THRESHOLD?: string;
}
