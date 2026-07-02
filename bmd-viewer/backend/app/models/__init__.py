"""ORM 도메인 모델.

도메인 경계:
  Doctor (사용자)  --(1:N)-->  Patient (환자)
  Patient          --(1:N)-->  XrayStudy (업로드된 X-ray 1건)
  XrayStudy        --(1:1)-->  BmdMeasurement (추론 결과)
  BmdMeasurement   --(1:N)-->  VertebraSegment (추출된 척추 분할)
  BmdMeasurement   --(1:N)-->  XaiFactor (설명성: 값에 영향 준 항목)

다른 프로젝트 통합을 고려해 테이블명에 prefix 없이 일반 명사를 사용하되,
혼선이 우려되면 __tablename__만 일괄 수정하면 된다.
"""
from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import (
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

    # DICOM 메타데이터 (촬영일 등)
    acquired_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    modality: Mapped[str | None] = mapped_column(String(16), nullable=True)
    # 추가 DICOM 태그 (JSON 문자열로 저장: 환자정보/장비/해상도 등)
    dicom_meta: Mapped[str | None] = mapped_column(Text, nullable=True)

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
