"""API 입출력 스키마 (Pydantic v2).

ORM 모델과 분리하여 API 계약을 명시적으로 관리한다.
프론트엔드 타입(lib/types)과 1:1로 대응하도록 유지한다.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field

from app.models import StudyStatus, UserRole


# ---------- 공통 ----------
class ORMBase(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# ---------- 인증 ----------
class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class DoctorCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)
    full_name: str
    role: UserRole = UserRole.DOCTOR


class DoctorOut(ORMBase):
    id: str
    email: EmailStr
    full_name: str
    role: UserRole
    created_at: datetime


# ---------- 환자 ----------
class PatientCreate(BaseModel):
    medical_record_no: str
    full_name: str
    birth_date: datetime | None = None
    sex: str | None = None
    notes: str | None = None


class PatientUpdate(BaseModel):
    full_name: str | None = None
    birth_date: datetime | None = None
    sex: str | None = None
    notes: str | None = None


class PatientOut(ORMBase):
    id: str
    medical_record_no: str
    full_name: str
    birth_date: datetime | None
    sex: str | None
    notes: str | None
    created_at: datetime


class PatientListItem(ORMBase):
    """리스트/검색 화면용 경량 표현."""

    id: str
    medical_record_no: str
    full_name: str
    sex: str | None
    # 최근 BMD 값 (목록에서 빠르게 확인)
    latest_bmd: float | None = None
    study_count: int = 0


# ---------- 척추 분할 / XAI ----------
class VertebraSegmentOut(ORMBase):
    label: str
    is_target: bool
    bbox_x: float
    bbox_y: float
    bbox_w: float
    bbox_h: float
    mask_path: str | None
    mean_intensity: float | None
    # v2: 모든 검출 척추(L1~L5)에 대해 산출 (예전엔 target(L4)만 BAR를 가졌음).
    bar: float | None = None
    qc_status: str | None = None
    qc_message: str | None = None
    # v3: 골소주 미세구조 텍스처 + AE 이상점수 (v18_spec.md Sec.5/C9).
    glcm_contrast: float | None = None
    glcm_correlation: float | None = None
    glcm_energy: float | None = None
    glcm_homogeneity: float | None = None
    glcm_entropy: float | None = None
    vario_slope: float | None = None
    fractal_dim: float | None = None
    anomaly_mse: float | None = None
    anomaly_pct: float | None = None
    anomaly_shap: dict | None = None
    # v4: 코호트 z-score/BHI (v18_spec.md Sec.5/D0). T-score 아님.
    bar_z: float | None = None
    bhi_z: float | None = None
    bhi_pct: float | None = None
    category: str | None = None


class XaiFactorOut(ORMBase):
    label: str
    contribution: float
    description: str | None
    rank: int


# ---------- BMD 측정 ----------
class BmdMeasurementOut(ORMBase):
    id: str
    target_vertebra: str
    bmd_value: float
    t_score: float | None
    z_score: float | None
    confidence: float | None
    exposure_corrected: bool
    gradcam_path: str | None
    l4_crop_path: str | None = None
    xai_overlay_path: str | None = None
    xai_l4_cam_path: str | None = None
    xai_bar_l1l5_path: str | None = None
    xai_air_soft_path: str | None = None
    # 밀도 히트맵 컬러 스케일 (파랑 끝 / 빨강 끝 / L4 평균 감쇠값)
    density_low: float | None = None
    density_high: float | None = None
    roi_mean_intensity: float | None = None
    # 측정 신뢰도 (경계 잘림 등). 값은 유지하되 경고 표시용.
    reliable: bool = True
    reliability_warning: str | None = None
    # 종적 대조용 L4 밀도 격자 (N×N, 0..1/None)
    l4_density_grid: list | None = None
    model_version: str | None
    computed_at: datetime
    segments: list[VertebraSegmentOut] = []
    xai_factors: list[XaiFactorOut] = []


# ---------- X-ray 스터디 ----------
class XrayStudyListItem(ORMBase):
    """환자 상세의 X-ray 이력 목록용."""

    id: str
    original_filename: str | None = None
    acquired_at: datetime | None
    uploaded_at: datetime
    status: StudyStatus
    modality: str | None
    preview_path: str | None
    bmd_value: float | None = None  # 완료된 경우만


class StudyNoteUpdate(BaseModel):
    note: str | None = None


class ViewOverrideIn(BaseModel):
    """의사가 화면에서 AP/LA를 수동 지정할 때의 요청 바디 (v18_spec.md Sec.4.5)."""

    view: Literal["AP", "LA"]


class XrayStudyDetail(ORMBase):
    """스터디 선택 시 상세 (원본 + 분할 + BMD)."""

    id: str
    patient_id: str
    dicom_path: str
    original_filename: str | None = None
    preview_path: str | None
    overlay_path: str | None = None  # 부분 실패 시 검출 오버레이
    acquired_at: datetime | None
    modality: str | None
    dicom_meta: str | None = None
    # 촬영 방향(v18_spec.md Sec.4.5). view_position: "AP"|"LA".
    # view_source: "auto"(DICOM 태그 자동판별) | "manual"(의사가 수동 지정).
    view_position: str | None = None
    view_source: str | None = None
    note: str | None = None
    note_updated_at: datetime | None = None
    status: StudyStatus
    error_message: str | None
    uploaded_at: datetime
    measurement: BmdMeasurementOut | None = None


# ---------- 추세 차트 ----------
class BmdTrendPoint(BaseModel):
    study_id: str
    measured_at: datetime
    bmd_value: float
    t_score: float | None = None


class BmdTrendOut(BaseModel):
    patient_id: str
    target_vertebra: str
    points: list[BmdTrendPoint]
    # 변화량 요약
    delta_absolute: float | None = None     # 최신 - 최초
    delta_percent: float | None = None
    # 측정 정밀도(LSC%) — 보정 이력이 있으면 최신값, 없으면 None(미보정).
    # |delta_percent| >= lsc_percent 여야 '재위치 노이즈가 아닌 실제 변화'.
    lsc_percent: float | None = None
    lsc_calibrated_at: datetime | None = None


# ---------- 측정 정밀도 (LSC) ----------
class PrecisionCalibrationOut(ORMBase):
    id: str
    n_studies: int
    n_repeats: int
    rms_cv_pct: float
    lsc_pct: float
    computed_at: datetime
