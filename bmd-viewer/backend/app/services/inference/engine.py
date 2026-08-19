"""BMD 추론 파이프라인 인터페이스.

업로드된 노트북(척추체 추출 + BMD 계산)의 실제 로직을 여기에 구현한다.
서비스 경계를 인터페이스로 고정해두면:
  - 노트북 코드를 BmdInferenceEngine 구현으로 이식하기만 하면 됨
  - 추후 다른 프로젝트로 추론 엔진만 분리/재사용 가능
  - 테스트 시 FakeEngine으로 대체 가능

⚠️ 현재는 인터페이스 + 스텁(stub) 구현이다.
   노트북이 제공되면 StubBmdEngine을 실제 구현으로 교체한다.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SegmentResult:
    label: str                       # "L1".."L5"
    is_target: bool                  # L4 여부
    bbox: tuple[float, float, float, float]  # (x, y, w, h) 정규화
    mask_path: str | None = None
    mean_intensity: float | None = None
    # v2: 모든 검출 척추에 대해 산출 (예전엔 target(L4)만 BAR를 가졌음).
    bar: float | None = None                 # Bone Attenuation Ratio (해당 척추)
    qc_status: str | None = None             # "PASS" | "WARN" | "FAIL"
    qc_message: str | None = None            # QC 사유 (여러 개면 "; "로 join)
    # v3: 골소주 미세구조 텍스처(노트북 v18 Sec.B4) + AE 이상점수(Sec.C9).
    glcm_contrast: float | None = None
    glcm_correlation: float | None = None
    glcm_energy: float | None = None
    glcm_homogeneity: float | None = None
    glcm_entropy: float | None = None
    vario_slope: float | None = None
    fractal_dim: float | None = None
    # 재구성오차(MSE)와 '정상' 코호트 대비 백분위. 특징이 하나라도 없으면(QC 등)
    # 둘 다 None — 임상 확정 진단이 아니라 코호트 상대적 이상 신호일 뿐이다.
    anomaly_mse: float | None = None
    anomaly_pct: float | None = None
    # SHAP(KernelExplainer): 이상점수에 각 특징이 기여한 정도 (v18 Sec.E3).
    # {feature_name: shap_value}. anomaly_pct가 None이면 이것도 None.
    anomaly_shap: dict | None = None
    # v4: 코호트 z-score/BHI (노트북 v17 §3 -> v18 D0). T-score가 아니라 이
    # 배포 코호트 내 상대 위치 — DXA 진단 아님(v18_spec.md Sec.4).
    bar_z: float | None = None       # BAR의 코호트 z-score
    bhi_z: float | None = None       # BAR+텍스처 가중합 복합 z-score
    bhi_pct: float | None = None     # 복합 z-score의 코호트 백분위(0-100)
    category: str | None = None      # WHO 컷포인트를 bar_z에 적용한 라벨


@dataclass
class XaiResult:
    label: str                       # 사람이 읽는 라벨
    contribution: float              # -1 ~ +1
    description: str | None = None


@dataclass
class InferenceResult:
    """추론 1건의 전체 산출물."""

    bmd_value: float
    target_vertebra: str = "L4"
    t_score: float | None = None
    z_score: float | None = None
    confidence: float | None = None
    exposure_corrected: bool = False
    model_version: str | None = None
    preview_path: str | None = None      # DICOM → PNG 변환 결과
    gradcam_path: str | None = None      # 설명성 오버레이
    l4_crop_path: str | None = None      # 추출된 L4 크롭 이미지
    xai_overlay_path: str | None = None  # 밀도 근거 히트맵 (L4 ROI 실제 감쇠)
    xai_l4_cam_path: str | None = None   # L4 크롭 Eigen-CAM 어트리뷰션
    xai_bar_l1l5_path: str | None = None  # L1~L5 전체 BAR 히트맵 (5-패널)
    xai_air_soft_path: str | None = None  # air/연부조직 기준값 근거 + BAR 계산식
    # 밀도 히트맵 컬러 스케일 (이미지별로 다름):
    #   density_low  = 파란색(저밀도) 끝 감쇠값
    #   density_high = 빨간색(고밀도) 끝 감쇠값
    #   roi_mean_attenuation = 이 L4 ROI 평균 감쇠값 (스케일 위 위치 = BMD)
    density_low: float | None = None
    density_high: float | None = None
    roi_mean_attenuation: float | None = None
    # 종적 대조용 L4 밀도 격자(N×N, 0..1 또는 None). post−pre로 골손실 부위 검출.
    l4_density_grid: list | None = None
    # 측정 신뢰도: 대상 척추가 이미지 경계에 잘렸거나(truncated) 부적합 영상이면
    # reliable=False, reliability_warning에 사유. 값은 유지하되 프론트에서 경고.
    reliable: bool = True
    reliability_warning: str | None = None
    segments: list[SegmentResult] = field(default_factory=list)
    xai_factors: list[XaiResult] = field(default_factory=list)
    acquired_at: str | None = None       # DICOM 메타에서 추출
    modality: str | None = None
    dicom_meta: dict | None = None       # 추가 DICOM 태그
    # 촬영 방향(v18_spec.md Sec.4.5) — soft_tissue_ref() 밴드 전략 선택에 쓰인다.
    view_position: str | None = None     # "AP" | "LA"
    view_source: str | None = None       # "auto" | "manual"


class PartialInferenceError(RuntimeError):
    """복구 가능한 추론 실패: L4 측정은 불가하나 원본/부분 추출물은 산출됨.

    이미 디스크에 저장된 부분 산출물 경로(preview/overlay)를 담아, 워커가
    이를 스터디에 기록해 화면에 '원본 + 추출한 척추'까지 보여주고, 실패한
    L4 분석 영역만 비워 실패 사유를 표시할 수 있게 한다.
    """

    def __init__(
        self,
        message: str,
        preview_path: str | None = None,
        overlay_path: str | None = None,
    ) -> None:
        super().__init__(message)
        self.preview_path = preview_path
        self.overlay_path = overlay_path


class BmdInferenceEngine(abc.ABC):
    """추론 엔진 추상 베이스.

    노트북 파이프라인 구현 시 이 클래스를 상속한다.
    단계는 노트북 흐름과 동일하게 구성:
      1) DICOM 로드 + 전처리 (노출 보정 포함)
      2) 척추 중심선/검출 (SpineNet/YOLO/Hourglass 등)
      3) L4 척추체 분할 (UNet++/EfficientNet-B4)
      4) proxy BMD 점수화
      5) Grad-CAM 설명성
      6) XAI 기여 항목 산출
    """

    @abc.abstractmethod
    def run(
        self,
        dicom_path: Path,
        output_dir: Path,
        view_override: str | None = None,
    ) -> InferenceResult:
        """단일 DICOM에 대해 전체 파이프라인을 실행한다.

        Args:
            dicom_path: 입력 DICOM 파일 경로
            output_dir: preview/mask/gradcam 등 파생 산출물 저장 디렉토리
            view_override: "AP"|"LA"를 명시하면 DICOM 태그 자동판별을 건너뛰고
                이 값을 그대로 쓴다 (의사가 뷰를 수동 지정해 재계산할 때).

        Returns:
            InferenceResult
        """
        raise NotImplementedError


class StubBmdEngine(BmdInferenceEngine):
    """노트북 이식 전까지 사용하는 스텁.

    파이프라인 배선/스키마/프론트 연동을 먼저 검증하기 위한 가짜 결과.
    실제 추론으로 교체하기 전까지만 사용한다.
    """

    MODEL_VERSION = "stub-0.1.0"

    def run(
        self,
        dicom_path: Path,
        output_dir: Path,
        view_override: str | None = None,
    ) -> InferenceResult:
        import random

        output_dir.mkdir(parents=True, exist_ok=True)
        bmd = round(random.uniform(0.6, 1.2), 3)

        segments = [
            SegmentResult(
                label=lbl,
                is_target=(lbl == "L4"),
                bbox=(0.4, 0.2 + i * 0.12, 0.2, 0.10),
                mean_intensity=round(random.uniform(0.3, 0.7), 3),
            )
            for i, lbl in enumerate(["L1", "L2", "L3", "L4", "L5"])
        ]
        xai = [
            XaiResult("척추체 피질골 밀도", 0.34, "L4 피질골 영역의 높은 감쇠"),
            XaiResult("골소주 패턴 균일도", 0.21, "내부 골소주 텍스처 기여"),
            XaiResult("노출/대비 보정", -0.09, "과노출 영역 보정으로 하향"),
            XaiResult("척추체 높이 비율", 0.12, "압박 골절 징후 없음"),
        ]
        return InferenceResult(
            bmd_value=bmd,
            target_vertebra="L4",
            t_score=round((bmd - 1.0) / 0.12, 2),
            confidence=round(random.uniform(0.7, 0.95), 2),
            exposure_corrected=True,
            model_version=self.MODEL_VERSION,
            segments=segments,
            xai_factors=xai,
            view_position=view_override or "AP",
            view_source="manual" if view_override else "auto",
        )


_engine_singleton: BmdInferenceEngine | None = None


def get_engine() -> BmdInferenceEngine:
    """현재 활성 추론 엔진을 반환한다 (싱글톤, 모델 1회 로드).

    BMD_INFERENCE_ENGINE=stub 이면 스텁을, 그 외에는 실제 YOLO 엔진을 사용한다.
    가중치가 없으면 자동으로 스텁으로 폴백한다.
    """
    global _engine_singleton
    if _engine_singleton is not None:
        return _engine_singleton

    import os

    which = os.getenv("BMD_INFERENCE_ENGINE", "yolo").lower()
    if which == "stub":
        _engine_singleton = StubBmdEngine()
        return _engine_singleton

    try:
        from app.services.inference.yolo_engine import YoloBmdEngine

        engine = YoloBmdEngine()
        if not engine._weights.exists():
            import logging

            logging.getLogger(__name__).warning(
                "모델 가중치 없음 → StubBmdEngine 폴백: %s", engine._weights
            )
            _engine_singleton = StubBmdEngine()
        else:
            _engine_singleton = engine
    except Exception as exc:  # noqa: BLE001
        import logging

        logging.getLogger(__name__).warning(
            "YOLO 엔진 초기화 실패 → StubBmdEngine 폴백: %s", exc
        )
        _engine_singleton = StubBmdEngine()

    return _engine_singleton
