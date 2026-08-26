// 데이터 제공자 인터페이스 (Repository 패턴).
// 컴포넌트는 이 인터페이스에만 의존한다.
// SQLite(REST) → Firebase 전환 시 이 인터페이스의 다른 구현만 갈아끼우면 된다.

import type {
  AuthSession,
  BmdTrend,
  ComparisonCreateInput,
  ComparisonType,
  Doctor,
  MultiPatientBmd,
  Patient,
  PatientListItem,
  PrecisionCalibration,
  SavedComparison,
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
  getCurrentDoctor(): Promise<Doctor>;

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
  updateStudyDate(studyId: string, acquiredAt: string): Promise<XrayStudyDetail>;
  overrideView(studyId: string, view: "AP" | "LA"): Promise<XrayStudyDetail>;

  // BMD 추세
  getBmdTrend(patientId: string): Promise<BmdTrend>;
  getMultiPatientBmd(patientIds: string[]): Promise<MultiPatientBmd[]>;

  // 측정 정밀도(LSC) 보정
  getLscCalibration(): Promise<PrecisionCalibration | null>;
  calibratePrecision(): Promise<PrecisionCalibration>;

  // 저장된 비교 (북마크)
  createComparison(input: ComparisonCreateInput): Promise<SavedComparison>;
  listComparisons(params?: {
    patientId?: string;
    type?: ComparisonType;
  }): Promise<SavedComparison[]>;
  deleteComparison(id: string): Promise<void>;

  // 정적 산출물(preview/gradcam/mask) URL 변환
  assetUrl(path: string | null): string | null;

  // 두 스터디의 L4 밀도 격자 차이를 matplotlib으로 렌더링한 PNG URL
  // (<img src>로 바로 사용, 인증 불필요 — assetUrl과 같은 보안 수준).
  densityDiffUrl(preStudyId: string, postStudyId: string): string;
}
