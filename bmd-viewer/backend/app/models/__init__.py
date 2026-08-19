"""ORM 도메인 모델.

도메인 경계:
  Doctor (사용자)  --(1:N)-->  Patient (환자)
  Patient          --(1:N)-->  XrayStudy (업로드된 X-ray 1건)
  XrayStudy        --(1:1)-->  BmdMeasurement (추론 결과)
  BmdMeasurement   --(1:N)-->  VertebraSegment (추출된 척추 분할)
  BmdMeasurement   --(1:N)-->  XaiFactor (설명성: 값에 영향 준 항목)

  PrecisionCalibration — 위 체인에 속하지 않는 독립 테이블. 특정 환자가 아닌
  '이 파이프라인의 측정 정밀도(LSC)'라는 시스템 속성이라 doctor/patient에
  종속되지 않는다.

다른 프로젝트 통합을 고려해 테이블명에 prefix 없이 일반 명사를 사용하되,
혼선이 우려되면 __tablename__만 일괄 수정하면 된다.
"""
from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    JSON,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


def _uuid() -> str:
    return str(uuid.uuid4())


class UserRole(str, enum.Enum):
    """역할 구분 (향후 확장 대비)."""

    ADMIN = "admin"
    DOCTOR = "doctor"
    RADIOLOGIST = "radiologist"
    VIEWER = "viewer"


class StudyStatus(str, enum.Enum):
    """X-ray 추론 처리 상태 (비동기 작업)."""

    UPLOADED = "uploaded"        # 업로드 완료, 추론 대기
    PROCESSING = "processing"    # 추론 진행 중
    COMPLETED = "completed"      # 추론 완료
    FAILED = "failed"            # 추론 실패


class Doctor(Base):
    __tablename__ = "doctors"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    full_name: Mapped[str] = mapped_column(String(120))
    role: Mapped[UserRole] = mapped_column(
        Enum(UserRole), default=UserRole.DOCTOR
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    patients: Mapped[list[Patient]] = relationship(
        back_populates="doctor", cascade="all, delete-orphan"
    )


class Patient(Base):
    __tablename__ = "patients"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    # 병원 내 환자 식별자 (MRN 등). 검색 대상.
    medical_record_no: Mapped[str] = mapped_column(
        String(64), index=True, unique=True
    )
    full_name: Mapped[str] = mapped_column(String(120), index=True)
    birth_date: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    sex: Mapped[str | None] = mapped_column(String(16), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    doctor_id: Mapped[str] = mapped_column(
        ForeignKey("doctors.id"), index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    doctor: Mapped[Doctor] = relationship(back_populates="patients")
    studies: Mapped[list[XrayStudy]] = relationship(
        back_populates="patient",
        cascade="all, delete-orphan",
        order_by="XrayStudy.acquired_at.desc()",
    )


class XrayStudy(Base):
    """업로드된 X-ray DICOM 1건과 그 처리 상태."""

    __tablename__ = "xray_studies"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    patient_id: Mapped[str] = mapped_column(
        ForeignKey("patients.id"), index=True
    )

    # 원본 DICOM 저장 경로 (storage/dicom 기준 상대경로)
    dicom_path: Mapped[str] = mapped_column(String(512))
    # 업로드된 원본 파일명 (화면 표시용)
    original_filename: Mapped[str | None] = mapped_column(
        String(512), nullable=True
    )
    # 화면 표시용 변환 이미지 (PNG) 경로
    preview_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    # 부분 실패(L4 측정 불가) 시 저장하는 검출 오버레이 경로 —
    # 실패한 스터디도 원본 + 추출한 척추까지 화면에 보여주기 위함.
    overlay_path: Mapped[str | None] = mapped_column(String(512), nullable=True)

    # DICOM 메타데이터 (촬영일 등)
    acquired_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    modality: Mapped[str | None] = mapped_column(String(16), nullable=True)
    # 추가 DICOM 태그 (JSON 문자열로 저장: 환자정보/장비/해상도 등)
    dicom_meta: Mapped[str | None] = mapped_column(Text, nullable=True)

    # 촬영 방향(AP/LA) — 연부조직 기준값 밴드 전략 선택에 쓰인다(v18_spec.md
    # Sec.4.5). view_position: "AP"|"LA". view_source: "auto"(DICOM 태그로
    # 자동 판별) | "manual"(태그가 없거나 틀려서 의사가 직접 지정 후 재계산).
    view_position: Mapped[str | None] = mapped_column(String(8), nullable=True)
    view_source: Mapped[str | None] = mapped_column(String(8), nullable=True)

    # 의사가 이 스터디에 작성하는 메모
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    # 메모 최종 작성/수정 일시
    note_updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    status: Mapped[StudyStatus] = mapped_column(
        Enum(StudyStatus), default=StudyStatus.UPLOADED, index=True
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    uploaded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    patient: Mapped[Patient] = relationship(back_populates="studies")
    measurement: Mapped[BmdMeasurement | None] = relationship(
        back_populates="study",
        cascade="all, delete-orphan",
        uselist=False,
    )


class BmdMeasurement(Base):
    """추론 결과: L4 기준 BMD 값과 부속 산출물."""

    __tablename__ = "bmd_measurements"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    study_id: Mapped[str] = mapped_column(
        ForeignKey("xray_studies.id"), index=True, unique=True
    )

    # 기준 척추 (요구사항: L4)
    target_vertebra: Mapped[str] = mapped_column(String(8), default="L4")
    # proxy BMD 점수 (모델 산출)
    bmd_value: Mapped[float] = mapped_column(Float)
    # T-score / Z-score (가능한 경우)
    t_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    z_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    # 모델 신뢰도 (0~1)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    # 노출/품질 보정 플래그
    exposure_corrected: Mapped[bool] = mapped_column(default=False)

    # Grad-CAM 등 설명성 오버레이 이미지 경로
    gradcam_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    # 추출된 L4 척추체 크롭 이미지 경로 (크게 보여주기용)
    l4_crop_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    # 밀도 근거 히트맵 (L4 ROI 실제 감쇠 색칠) 이미지 경로
    xai_overlay_path: Mapped[str | None] = mapped_column(
        String(512), nullable=True
    )
    # L4 크롭 Eigen-CAM 어트리뷰션 이미지 경로
    xai_l4_cam_path: Mapped[str | None] = mapped_column(
        String(512), nullable=True
    )
    # L1~L5 전체 BAR 히트맵 (5-패널, 노트북 v18 E2) 이미지 경로
    xai_bar_l1l5_path: Mapped[str | None] = mapped_column(
        String(512), nullable=True
    )
    # air/연부조직 기준값이 실제로 어느 픽셀에서 왔는지 + BAR 계산식 이미지 경로
    xai_air_soft_path: Mapped[str | None] = mapped_column(
        String(512), nullable=True
    )

    # 밀도 히트맵 컬러 스케일 (이미지별로 다름). 프론트에서 파랑/빨강 끝
    # 값과 L4 평균 위치를 눈금으로 표시하는 데 사용.
    density_low: Mapped[float | None] = mapped_column(Float, nullable=True)
    density_high: Mapped[float | None] = mapped_column(Float, nullable=True)
    roi_mean_intensity: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )

    # 측정 신뢰도. 대상 척추가 이미지 경계에 잘렸거나 부적합 영상이면 False,
    # 사유는 reliability_warning. 값은 유지하되 프론트에서 경고 표시.
    reliable: Mapped[bool] = mapped_column(default=True)
    reliability_warning: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )
    # 종적 대조용 L4 밀도 격자(N×N, 0..1/None). SQLite엔 JSON(TEXT)로 저장.
    l4_density_grid: Mapped[list | None] = mapped_column(JSON, nullable=True)

    model_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    study: Mapped[XrayStudy] = relationship(back_populates="measurement")
    segments: Mapped[list[VertebraSegment]] = relationship(
        back_populates="measurement", cascade="all, delete-orphan"
    )
    xai_factors: Mapped[list[XaiFactor]] = relationship(
        back_populates="measurement", cascade="all, delete-orphan"
    )


class VertebraSegment(Base):
    """추출된 척추 1개의 분할 마스크/박스 정보."""

    __tablename__ = "vertebra_segments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    measurement_id: Mapped[str] = mapped_column(
        ForeignKey("bmd_measurements.id"), index=True
    )

    label: Mapped[str] = mapped_column(String(8))  # L1~L5 등
    is_target: Mapped[bool] = mapped_column(default=False)  # L4 여부

    # bounding box (정규화 좌표 0~1)
    bbox_x: Mapped[float] = mapped_column(Float)
    bbox_y: Mapped[float] = mapped_column(Float)
    bbox_w: Mapped[float] = mapped_column(Float)
    bbox_h: Mapped[float] = mapped_column(Float)

    # 분할 마스크 경로 (PNG/RLE) - 오버레이용
    mask_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    # 해당 척추의 평균 픽셀 강도 등 보조값
    mean_intensity: Mapped[float | None] = mapped_column(Float, nullable=True)
    # v2: 모든 검출 척추(L1~L5)에 대해 산출 (예전엔 target(L4)만 BAR를 가졌음).
    bar: Mapped[float | None] = mapped_column(Float, nullable=True)
    qc_status: Mapped[str | None] = mapped_column(String(8), nullable=True)
    qc_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # v3: 골소주 미세구조 텍스처(노트북 v18 Sec.B4) + AE 이상점수(Sec.C9).
    glcm_contrast: Mapped[float | None] = mapped_column(Float, nullable=True)
    glcm_correlation: Mapped[float | None] = mapped_column(Float, nullable=True)
    glcm_energy: Mapped[float | None] = mapped_column(Float, nullable=True)
    glcm_homogeneity: Mapped[float | None] = mapped_column(Float, nullable=True)
    glcm_entropy: Mapped[float | None] = mapped_column(Float, nullable=True)
    vario_slope: Mapped[float | None] = mapped_column(Float, nullable=True)
    fractal_dim: Mapped[float | None] = mapped_column(Float, nullable=True)
    anomaly_mse: Mapped[float | None] = mapped_column(Float, nullable=True)
    anomaly_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    # SHAP(KernelExplainer): 이상점수에 각 특징이 기여한 정도 (v18 Sec.E3).
    # {feature_name: shap_value}. SQLite엔 JSON(TEXT)로 저장.
    anomaly_shap: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # v4: 코호트 z-score/BHI (노트북 v17 §3 -> v18 D0). T-score 아님 — 이
    # 배포 코호트 내 상대 위치일 뿐(v18_spec.md Sec.4 정직성 원칙).
    bar_z: Mapped[float | None] = mapped_column(Float, nullable=True)
    bhi_z: Mapped[float | None] = mapped_column(Float, nullable=True)
    bhi_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    category: Mapped[str | None] = mapped_column(String(64), nullable=True)

    measurement: Mapped[BmdMeasurement] = relationship(
        back_populates="segments"
    )


class XaiFactor(Base):
    """설명성: BMD 값에 영향을 준 항목과 기여도.

    의사가 이해할 수 있도록 사람이 읽을 수 있는 라벨 + 기여 방향/크기를 저장.
    예: '척추체 피질골 밀도' +0.32, '노출 과다 보정' -0.08
    """

    __tablename__ = "xai_factors"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    measurement_id: Mapped[str] = mapped_column(
        ForeignKey("bmd_measurements.id"), index=True
    )

    # 사람이 읽을 라벨
    label: Mapped[str] = mapped_column(String(120))
    # 기여도 (-1 ~ +1, 부호는 BMD를 높이거나 낮춘 방향)
    contribution: Mapped[float] = mapped_column(Float)
    # 추가 설명 텍스트
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    rank: Mapped[int] = mapped_column(Integer, default=0)

    measurement: Mapped[BmdMeasurement] = relationship(
        back_populates="xai_factors"
    )


class PrecisionCalibration(Base):
    """측정 정밀도(LSC) 보정 결과 — 재위치 섭동 재측정으로 얻은 RMS-CV%/LSC%.

    특정 환자가 아니라 '이 파이프라인이 얼마나 정밀한가'라는 시스템 속성이므로
    doctor/patient에 종속시키지 않는다. 매 보정마다 새 행을 추가(감사 이력),
    조회는 항상 최신 행(computed_at DESC LIMIT 1)을 사용한다.
    """

    __tablename__ = "precision_calibrations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    n_studies: Mapped[int] = mapped_column(Integer)
    n_repeats: Mapped[int] = mapped_column(Integer)
    rms_cv_pct: Mapped[float] = mapped_column(Float)
    lsc_pct: Mapped[float] = mapped_column(Float)
    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
