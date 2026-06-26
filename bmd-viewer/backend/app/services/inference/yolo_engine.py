"""YOLOv8 segmentation 기반 실제 BMD 추론 엔진.

노트북(L4_AP_segmentation.ipynb)의 파이프라인을 백엔드 서비스로 이식한 것이다.

파이프라인:
  1) DICOM 로드 + 전처리 (MONOCHROME1 보정, Rescale 적용)
     - 모델 입력용 8bit RGB (노출 정규화, 뼈 밝게)
     - 골밀도 계산용 dens 배열 (선형성 보존, percentile 클립 없음)
  2) YOLOv8-seg 추론 → L1~L5 분할
  3) L4 선택 (class 직접 매칭, 실패 시 위→아래 4번째 fallback)
  4) L4 폴리곤 → 마스크 (ROI_SHRINK으로 중앙 해면골 영역만)
  5) 마스크 영역 dens 픽셀 통계 = proxy BMD
  6) 분할 오버레이/미리보기 저장 + XAI 기여 항목 산출
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from app.core.config import settings
from app.services.inference.engine import (
    BmdInferenceEngine,
    InferenceResult,
    SegmentResult,
    XaiResult,
)

logger = logging.getLogger(__name__)

CLASS_NAMES = ["L1", "L2", "L3", "L4", "L5"]
L4_CLASS_ID = CLASS_NAMES.index("L4")  # 3

# 노트북 하이퍼파라미터
INVERT = True          # MONOCHROME1을 학습 톤에 맞춰 뼈를 밝게
CONF = 0.25            # 검출 신뢰도 임계
ROI_SHRINK = 0.7       # 1.0=척추체 전체, 0.7=중앙 해면골 ROI

# proxy BMD 정규화 기준 (dens 평균값을 0~1 BMD 점수로 환산)
# 데이터셋 분포에 맞춰 보정 가능. 노트북 산출 dens는 raw 강도라 스케일이 크다.
_BMD_REF_LOW = 0.0
_BMD_REF_HIGH = 4096.0


class YoloBmdEngine(BmdInferenceEngine):
    """YOLOv8-seg 기반 L1~L5 분할 + L4 proxy BMD 산출 엔진."""

    MODEL_VERSION = "yolov8-seg-l1l5-1.0"

    def __init__(self, weights_path: Path | None = None) -> None:
        self._weights = weights_path or (settings.ml_model_dir / "l1_l5_seg.pt")
        self._model = None  # lazy load

    def _ensure_model(self):
        if self._model is None:
            from ultralytics import YOLO

            if not self._weights.exists():
                raise FileNotFoundError(
                    f"모델 가중치를 찾을 수 없습니다: {self._weights}"
                )
            logger.info("YOLO 모델 로드: %s", self._weights)
            self._model = YOLO(str(self._weights))
        return self._model

    # ---------- 노트북 함수 이식 ----------
    @staticmethod
    def _load_dicom(path: Path):
        """DICOM 로드. (model_rgb uint8, dens float, H, W) 반환."""
        import pydicom

        ds = pydicom.dcmread(str(path))
        raw = ds.pixel_array.astype(np.float32)
        raw = raw * float(getattr(ds, "RescaleSlope", 1.0)) + float(
            getattr(ds, "RescaleIntercept", 0.0)
        )
        mono1 = str(
            getattr(ds, "PhotometricInterpretation", "")
        ).upper() == "MONOCHROME1"

        # 골밀도용: 값이 클수록 밀도 높게 방향만 표준화 (선형성 보존)
        dens = (raw.max() - raw) if mono1 else raw.copy()

        # 모델 입력용: 8bit 정규화
        a = raw.copy()
        if INVERT and mono1:
            a = a.max() - a
        lo, hi = np.percentile(a, 0.5), np.percentile(a, 99.5)
        if hi <= lo:
            lo, hi = a.min(), a.max() + 1e-6
        rgb = cv2.cvtColor(
            (np.clip((a - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8),
            cv2.COLOR_GRAY2RGB,
        )
        H, W = raw.shape
        acquired = getattr(ds, "AcquisitionDate", None) or getattr(
            ds, "StudyDate", None
        )
        modality = str(getattr(ds, "Modality", "") or "") or None
        return rgb, dens, H, W, acquired, modality

    @staticmethod
    def _poly_to_mask(poly_xy, H, W, shrink=1.0):
        poly = np.asarray(poly_xy, dtype=np.float32)
        if shrink != 1.0 and len(poly):
            ctr = poly.mean(0)
            poly = ctr + shrink * (poly - ctr)
        m = np.zeros((H, W), np.uint8)
        cv2.fillPoly(m, [poly.astype(np.int32)], 1)
        return m

    @staticmethod
    def _pick_l4_index(boxes_xyxy, classes, confs):
        """L4 선택: class 직접 매칭 우선, 실패 시 위→아래 4번째."""
        if len(boxes_xyxy) == 0:
            return None, "검출 없음"
        idx = [i for i, c in enumerate(classes) if c == L4_CLASS_ID]
        if idx:
            return max(idx, key=lambda i: confs[i]), "L4 클래스 직접 검출"
        order = sorted(
            range(len(boxes_xyxy)),
            key=lambda i: (boxes_xyxy[i][1] + boxes_xyxy[i][3]) / 2,
        )
        if len(order) >= 4:
            return order[3], "위→아래 4번째 (Fallback)"
        return None, f"척추 {len(boxes_xyxy)}개만 검출"

    @staticmethod
    def _bbox_norm(xyxy, W, H):
        x1, y1, x2, y2 = xyxy
        return (
            float(x1 / W),
            float(y1 / H),
            float((x2 - x1) / W),
            float((y2 - y1) / H),
        )

    def _bmd_score(self, mean_dens: float) -> float:
        """dens 평균을 0~1 proxy BMD 점수로 환산."""
        score = (mean_dens - _BMD_REF_LOW) / (_BMD_REF_HIGH - _BMD_REF_LOW)
        return round(float(np.clip(score, 0.0, 1.5)), 3)

    # ---------- 메인 ----------
    def run(self, dicom_path: Path, output_dir: Path) -> InferenceResult:
        model = self._ensure_model()
        output_dir.mkdir(parents=True, exist_ok=True)

        rgb, dens, H, W, acquired, modality = self._load_dicom(dicom_path)

        # 미리보기 저장
        preview_name = "preview.png"
        cv2.imwrite(
            str(output_dir / preview_name),
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        )

        r = model.predict(rgb, conf=CONF, verbose=False)[0]
        if r.masks is None or len(r.boxes) == 0:
            raise RuntimeError("척추가 검출되지 않았습니다 (영상 품질/노출 확인)")

        boxes = r.boxes.xyxy.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()

        l4_i, status = self._pick_l4_index(boxes, classes, confs)
        if l4_i is None:
            raise RuntimeError(f"L4를 특정할 수 없습니다: {status}")

        # 검출된 모든 척추를 분할 결과로 기록
        segments: list[SegmentResult] = []
        for i in range(len(boxes)):
            cid = classes[i]
            label = CLASS_NAMES[cid] if 0 <= cid < len(CLASS_NAMES) else str(cid)
            m_full = self._poly_to_mask(r.masks.xy[i], H, W, shrink=1.0)
            seg_vals = dens[m_full > 0]
            segments.append(
                SegmentResult(
                    label=label,
                    is_target=(i == l4_i),
                    bbox=self._bbox_norm(boxes[i], W, H),
                    mean_intensity=(
                        round(float(seg_vals.mean()), 2)
                        if seg_vals.size
                        else None
                    ),
                )
            )

        # L4 ROI (중앙 해면골) 골밀도 통계
        mask = self._poly_to_mask(r.masks.xy[l4_i], H, W, shrink=ROI_SHRINK)
        vals = dens[mask > 0]
        if vals.size == 0:
            raise RuntimeError("L4 마스크가 비어 있습니다")

        mean_d = float(vals.mean())
        median_d = float(np.median(vals))
        std_d = float(vals.std())
        p10 = float(np.percentile(vals, 10))
        p90 = float(np.percentile(vals, 90))
        bmd_value = self._bmd_score(mean_d)

        # 분할 오버레이 저장 (L4 녹색)
        vis = rgb.copy()
        vis[mask > 0] = (
            0.5 * vis[mask > 0] + 0.5 * np.array([0, 255, 0])
        ).astype(np.uint8)
        overlay_name = "l4_overlay.png"
        cv2.imwrite(
            str(output_dir / overlay_name),
            cv2.cvtColor(vis, cv2.COLOR_RGB2BGR),
        )

        # L4 마스크 npy 저장 (재현/감사용)
        np.save(str(output_dir / "l4_mask.npy"), mask)

        # XAI 기여 항목 산출 (통계 기반 해석)
        xai = self._build_xai(
            mean_d, median_d, std_d, p10, p90, float(confs[l4_i])
        )

        # 신뢰도 = L4 검출 confidence
        confidence = round(float(confs[l4_i]), 3)

        # 촬영일 파싱 (YYYYMMDD)
        acquired_at = None
        if acquired and len(str(acquired)) == 8:
            s = str(acquired)
            acquired_at = f"{s[:4]}-{s[4:6]}-{s[6:8]}T00:00:00"

        return InferenceResult(
            bmd_value=bmd_value,
            target_vertebra="L4",
            t_score=None,  # 참조군 분포가 있으면 산출. 현재 미제공.
            confidence=confidence,
            exposure_corrected=True,
            model_version=self.MODEL_VERSION,
            preview_path=f"{output_dir.name}/{preview_name}",
            gradcam_path=f"{output_dir.name}/{overlay_name}",
            segments=segments,
            xai_factors=xai,
            acquired_at=acquired_at,
            modality=modality,
        )

    @staticmethod
    def _build_xai(mean_d, median_d, std_d, p10, p90, conf) -> list[XaiResult]:
        """L4 ROI 통계로부터 의사가 이해할 XAI 기여 항목을 구성.

        통계량을 사람이 읽을 수 있는 근거로 변환한다.
        contribution은 -1~+1 정규화된 상대 기여(해석 보조용).
        """
        spread = std_d / (abs(mean_d) + 1e-6)  # 변동계수
        skew = (mean_d - median_d) / (abs(median_d) + 1e-6)
        dynamic = (p90 - p10) / (abs(mean_d) + 1e-6)

        factors = [
            XaiResult(
                label="L4 ROI 평균 감쇠",
                contribution=round(float(np.clip(mean_d / 4096.0, -1, 1)), 3),
                description=f"중앙 해면골 영역 평균 강도 {mean_d:.1f}",
            ),
            XaiResult(
                label="밀도 균일도",
                contribution=round(float(np.clip(-spread, -1, 1)), 3),
                description=f"변동계수 {spread:.3f} (낮을수록 균일)",
            ),
            XaiResult(
                label="강도 분포 동적범위",
                contribution=round(float(np.clip(-dynamic * 0.5, -1, 1)), 3),
                description=f"p10–p90 폭 {p90 - p10:.1f}",
            ),
            XaiResult(
                label="분포 비대칭",
                contribution=round(float(np.clip(skew, -1, 1)), 3),
                description=f"평균-중앙값 편차 {skew:+.3f}",
            ),
            XaiResult(
                label="L4 검출 신뢰도",
                contribution=round(float(np.clip(conf, -1, 1)), 3),
                description=f"분할 모델 confidence {conf:.2f}",
            ),
        ]
        # 절대 기여 크기 순 정렬
        factors.sort(key=lambda f: abs(f.contribution), reverse=True)
        return factors
