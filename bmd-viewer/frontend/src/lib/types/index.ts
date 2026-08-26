// 백엔드 스키마와 1:1 대응하는 도메인 타입.
// 데이터 제공자(REST/Firebase)와 무관하게 컴포넌트가 사용하는 단일 계약.

export type StudyStatus = "uploaded" | "processing" | "completed" | "failed";
export type UserRole = "admin" | "doctor" | "radiologist" | "viewer";

export interface Doctor {
  id: string;
  email: string;
  full_name: string;
  role: UserRole;
  created_at: string;
}

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
  // v2: populated for EVERY detected vertebra (used to be target(L4)-only).
  bar: number | null;
  qc_status: "PASS" | "WARN" | "FAIL" | null;
  qc_message: string | null;
  // v3: trabecular microarchitecture texture (GLCM/variogram/fractal) + AE
  // reconstruction-error anomaly score (v18_spec.md Sec.5/C9). Cohort-relative,
  // not a diagnosis -- null whenever any input feature is unavailable (e.g. ROI
  // too small for a stable fractal-dimension estimate).
  glcm_contrast: number | null;
  glcm_correlation: number | null;
  glcm_energy: number | null;
  glcm_homogeneity: number | null;
  glcm_entropy: number | null;
  vario_slope: number | null;
  fractal_dim: number | null;
  anomaly_mse: number | null;
  anomaly_pct: number | null;
  // SHAP (KernelExplainer): how much each feature pushed the anomaly score up
  // (+) or down (-) for this vertebra (v18_spec.md Sec.5/E3). null whenever
  // anomaly_pct is null.
  anomaly_shap: Record<string, number> | null;
  // v4: cohort z-score / BHI (v18_spec.md Sec.5/D0). NOT a DXA T-score --
  // relative position within this deployment's own reference cohort only.
  bar_z: number | null;
  bhi_z: number | null;
  bhi_pct: number | null;
  category: string | null;
}

// 5구역 종적 대조가 공유하는 구역 이름 — 밀도(프론트 재집계)와 텍스처
// (백엔드 계산) 양쪽에서 동일한 의미로 쓰인다.
export type CompareRegionName =
  | "center"
  | "top_left"
  | "top_right"
  | "bottom_left"
  | "bottom_right";

export interface TextureRegionFeatures {
  glcm_homogeneity: number | null;
  glcm_entropy: number | null;
  vario_slope: number | null;
  fractal_dim: number | null;
}

export type TextureRegions = Partial<
  Record<CompareRegionName, TextureRegionFeatures | null>
>;

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
  xai_bar_l1l5_path: string | null;
  xai_air_soft_path: string | null;
  // 밀도 히트맵 컬러 스케일 (이미지별로 다름):
  // density_low = 파란색(저밀도) 끝, density_high = 빨간색(고밀도) 끝,
  // roi_mean_intensity = 이 L4 평균 감쇠값 (스케일 위 위치)
  density_low: number | null;
  density_high: number | null;
  roi_mean_intensity: number | null;
  // 측정 신뢰도: 대상 척추가 이미지 경계에 잘렸는지 등. false면 경고 표시.
  reliable: boolean;
  reliability_warning: string | null;
  // 종적 대조용 L4 밀도 격자 (N×N, 0..1/null). post−pre로 골손실 부위 검출.
  l4_density_grid: (number | null)[][] | null;
  // 위 격자의 실제 L4 ROI bbox 가로/세로 비율 (width/height) -- 격자를
  // 정사각형이 아니라 실제 ROI 비율로 그리기 위함. N×N 셀 개수와는 무관.
  l4_density_grid_aspect: number | null;
  // 종적 대조(텍스처, "detailed" 모드)용 5구역 텍스처 특징. 셀 격자는
  // GLCM/fractal/variogram에는 너무 잘게 쪼개져 무의미해지므로, 처음부터
  // 중앙+네 모서리 5구역으로 계산된다. 표본이 너무 작은 구역은 null.
  l4_texture_regions: TextureRegions | null;
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
  overlay_path: string | null; // 부분 실패 시 검출 오버레이
  acquired_at: string | null;
  modality: string | null;
  // 촬영 방향(v18 Sec.4.5): view_position "AP"|"LA", view_source "auto"|"manual".
  // 자동판별 태그가 없으면 AP를 기본으로 쓰고(view_source="auto"), 자동판별이
  // 틀렸다면 의사가 화면에서 수동 지정해 재계산할 수 있다(view_source="manual").
  view_position: "AP" | "LA" | null;
  view_source: "auto" | "manual" | null;
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
  // 측정 정밀도(LSC%) — 보정 이력이 있으면 최신값, 없으면 null(미보정).
  lsc_percent: number | null;
  lsc_calibrated_at: string | null;
}

// ---------- 저장된 비교 ----------
export type ComparisonType = "pre_post" | "multi_patient";

export interface SavedComparison {
  id: string;
  type: ComparisonType;
  title: string;
  patient_id: string | null;
  pre_study_id: string | null;
  post_study_ids: string[] | null;
  patient_ids: string[] | null;
  config: Record<string, unknown> | null;
  created_at: string;
}

export interface ComparisonCreateInput {
  type: ComparisonType;
  title: string;
  patient_id?: string | null;
  pre_study_id?: string | null;
  post_study_ids?: string[] | null;
  patient_ids?: string[] | null;
  config?: Record<string, unknown> | null;
}

// ---------- 다중 환자 비교 ----------
export interface MultiPatientBmd {
  patient_id: string;
  patient_name: string;
  target_vertebra: string;
  points: BmdTrendPoint[];
  delta_percent: number | null;
}

export interface PrecisionCalibration {
  id: string;
  n_studies: number;
  n_repeats: number;
  rms_cv_pct: number;
  lsc_pct: number;
  computed_at: string;
}

export interface AuthSession {
  access_token: string;
}
