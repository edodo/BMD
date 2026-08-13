// 데이터 제공자 인터페이스 (Repository 패턴).
// 컴포넌트는 이 인터페이스에만 의존한다.
// SQLite(REST) → Firebase 전환 시 이 인터페이스의 다른 구현만 갈아끼우면 된다.

import type {
  AuthSession,
  BmdTrend,
  Patient,
  PatientListItem,
  PrecisionCalibration,
  XrayStudyDetail,
  XrayStudyListItem,
} from "@/lib/types";

export interface DataProvider {
  // 인증
  login(email: string, password: string): Promise<AuthSession>;
  register(input: {
    email: string;
    password: string;
    full_name: string;
    role?: string;
  }): Promise<void>;
  logout(): void;
  isAuthenticated(): boolean;

  // 환자
  listPatients(query?: string): Promise<PatientListItem[]>;
  getPatient(id: string): Promise<Patient>;
  createPatient(input: Partial<Patient>): Promise<Patient>;
  deletePatient(id: string): Promise<void>;

  // X-ray 스터디
  listStudies(patientId: string): Promise<XrayStudyListItem[]>;
  uploadStudy(patientId: string, file: File): Promise<XrayStudyDetail>;
  getStudy(studyId: string): Promise<XrayStudyDetail>;
  deleteStudy(studyId: string): Promise<void>;
  updateStudyNote(studyId: string, note: string): Promise<XrayStudyDetail>;
  overrideView(studyId: string, view: "AP" | "LA"): Promise<XrayStudyDetail>;

  // BMD 추세
  getBmdTrend(patientId: string): Promise<BmdTrend>;

  // 측정 정밀도(LSC) 보정
  getLscCalibration(): Promise<PrecisionCalibration | null>;
  calibratePrecision(): Promise<PrecisionCalibration>;

  // 정적 산출물(preview/gradcam/mask) URL 변환
  assetUrl(path: string | null): string | null;
}
