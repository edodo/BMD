// 백엔드 스키마와 1:1 대응하는 도메인 타입.
// 데이터 제공자(REST/Firebase)와 무관하게 컴포넌트가 사용하는 단일 계약.

export type StudyStatus = "uploaded" | "processing" | "completed" | "failed";
export type UserRole = "admin" | "doctor" | "radiologist" | "viewer";

export interface Patient {
  id: string;
  medical_record_no: string;
  full_name: string;
  birth_date: string | null;
  sex: string | null;
  notes: string | null;
  created_at: string;
}

export interface PatientListItem {
  id: string;
  medical_record_no: string;
  full_name: string;
  sex: string | null;
  latest_bmd: number | null;
  study_count: number;
}

export interface VertebraSegment {
  label: string;
  is_target: boolean;
  bbox_x: number;
  bbox_y: number;
  bbox_w: number;
  bbox_h: number;
  mask_path: string | null;
  mean_intensity: number | null;
}

export interface XaiFactor {
  label: string;
  contribution: number; // -1 ~ +1
  description: string | null;
  rank: number;
}

export interface BmdMeasurement {
  id: string;
  target_vertebra: string; // "L4"
  bmd_value: number;
  t_score: number | null;
  z_score: number | null;
  confidence: number | null;
  exposure_corrected: boolean;
  gradcam_path: string | null;
  model_version: string | null;
  computed_at: string;
  segments: VertebraSegment[];
  xai_factors: XaiFactor[];
}

export interface XrayStudyListItem {
  id: string;
  acquired_at: string | null;
  uploaded_at: string;
  status: StudyStatus;
  modality: string | null;
  preview_path: string | null;
  bmd_value: number | null;
}

export interface XrayStudyDetail {
  id: string;
  patient_id: string;
  dicom_path: string;
  preview_path: string | null;
  acquired_at: string | null;
  modality: string | null;
  status: StudyStatus;
  error_message: string | null;
  uploaded_at: string;
  measurement: BmdMeasurement | null;
}

export interface BmdTrendPoint {
  study_id: string;
  measured_at: string;
  bmd_value: number;
  t_score: number | null;
}

export interface BmdTrend {
  patient_id: string;
  target_vertebra: string;
  points: BmdTrendPoint[];
  delta_absolute: number | null;
  delta_percent: number | null;
}

export interface AuthSession {
  access_token: string;
}
