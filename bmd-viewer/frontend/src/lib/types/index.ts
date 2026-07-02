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
  l4_crop_path: string | null;
  xai_overlay_path: string | null;
  xai_l4_cam_path: string | null;
  // 밀도 히트맵 컬러 스케일 (이미지별로 다름):
  // density_low = 파란색(저밀도) 끝, density_high = 빨간색(고밀도) 끝,
  // roi_mean_intensity = 이 L4 평균 감쇠값 (스케일 위 위치)
  density_low: number | null;
  density_high: number | null;
  roi_mean_intensity: number | null;
  // 측정 신뢰도: 대상 척추가 이미지 경계에 잘렸는지 등. false면 경고 표시.
  reliable: boolean;
  reliability_warning: string | null;
  model_version: string | null;
  computed_at: string;
  segments: VertebraSegment[];
  xai_factors: XaiFactor[];
}

export interface XrayStudyListItem {
  id: string;
  original_filename: string | null;
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
  original_filename: string | null;
  dicom_meta: string | null;
  note: string | null;
  note_updated_at: string | null;
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
