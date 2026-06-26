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
CONF = 0.25            # detection confidence threshold
ROI_SHRINK = 0.7       # 1.0=whole vertebral body, 0.7=central cancellous ROI

# BMD is normalized per-image (relative to this image's own soft-tissue
# baseline and bone peak), so no fixed reference constant is used.
# This keeps the score stable across different DICOM bit depths/exposures.


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
                    f"Model weights not found: {self._weights}"
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
        acquired_time = getattr(ds, "AcquisitionTime", None) or getattr(
            ds, "StudyTime", None
        )
        modality = str(getattr(ds, "Modality", "") or "") or None

        # Collect human-readable DICOM tags for display (skip pixel data)
        def _tag(name):
            v = getattr(ds, name, None)
            return str(v) if v not in (None, "") else None

        meta = {
            "PatientID": _tag("PatientID"),
            "PatientSex": _tag("PatientSex"),
            "PatientAge": _tag("PatientAge"),
            "StudyDate": _tag("StudyDate"),
            "AcquisitionDate": _tag("AcquisitionDate"),
            "Modality": modality,
            "Manufacturer": _tag("Manufacturer"),
            "ManufacturerModelName": _tag("ManufacturerModelName"),
            "BodyPartExamined": _tag("BodyPartExamined"),
            "ViewPosition": _tag("ViewPosition"),
            "KVP": _tag("KVP"),
            "Exposure": _tag("Exposure"),
            "Rows": int(getattr(ds, "Rows", H) or H),
            "Columns": int(getattr(ds, "Columns", W) or W),
            "BitsStored": _tag("BitsStored"),
            "PhotometricInterpretation": _tag("PhotometricInterpretation"),
        }
        # Drop empty entries
        meta = {k: v for k, v in meta.items() if v is not None}
        return rgb, dens, H, W, acquired, acquired_time, modality, meta

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
            return None, "No detection"
        idx = [i for i, c in enumerate(classes) if c == L4_CLASS_ID]
        if idx:
            return max(idx, key=lambda i: confs[i]), "L4 class direct match"
        order = sorted(
            range(len(boxes_xyxy)),
            key=lambda i: (boxes_xyxy[i][1] + boxes_xyxy[i][3]) / 2,
        )
        if len(order) >= 4:
            return order[3], "Top-to-bottom 4th (fallback)"
        return None, f"Only {len(boxes_xyxy)} vertebrae detected"

    @staticmethod
    def _bbox_norm(xyxy, W, H):
        x1, y1, x2, y2 = xyxy
        return (
            float(x1 / W),
            float(y1 / H),
            float((x2 - x1) / W),
            float((y2 - y1) / H),
        )

    @staticmethod
    def _bmd_score(mean_dens: float, dens: np.ndarray, roi_vals: np.ndarray) -> float:
        """Convert L4 ROI mean intensity to a 0~1 proxy BMD score.

        Per-image relative normalization that is robust to differing DICOM
        bit depths and exposure. We anchor against this image's own
        intensity distribution:
          - low anchor  = 25th percentile of the whole image (soft tissue)
          - high anchor = 99th percentile of the whole image (densest bone)
        The L4 ROI mean is mapped into [low, high] → 0..1. A midpoint
        fallback keeps the score meaningful when anchors are degenerate
        (e.g. synthetic/binary images).
        """
        low = float(np.percentile(dens, 25))
        high = float(np.percentile(dens, 99))
        denom = high - low
        if denom <= 1e-6:
            # Degenerate distribution: fall back to ROI's own spread
            roi_lo = float(np.percentile(roi_vals, 5))
            roi_hi = float(np.percentile(roi_vals, 95))
            denom = max(roi_hi - roi_lo, 1e-6)
            low = roi_lo
        score = (mean_dens - low) / denom
        return round(float(np.clip(score, 0.0, 1.0)), 3)

    # ---------- 메인 ----------
    def run(self, dicom_path: Path, output_dir: Path) -> InferenceResult:
        model = self._ensure_model()
        output_dir.mkdir(parents=True, exist_ok=True)

        rgb, dens, H, W, acquired, acquired_time, modality, dicom_meta = (
            self._load_dicom(dicom_path)
        )

        # 미리보기 저장
        preview_name = "preview.png"
        cv2.imwrite(
            str(output_dir / preview_name),
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        )

        r = model.predict(rgb, conf=CONF, verbose=False)[0]
        if r.masks is None or len(r.boxes) == 0:
            raise RuntimeError("No vertebrae detected (check image quality/exposure)")

        boxes = r.boxes.xyxy.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()

        l4_i, status = self._pick_l4_index(boxes, classes, confs)
        if l4_i is None:
            raise RuntimeError(f"Could not identify L4: {status}")

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
            raise RuntimeError("L4 mask is empty")

        mean_d = float(vals.mean())
        median_d = float(np.median(vals))
        std_d = float(vals.std())
        p10 = float(np.percentile(vals, 10))
        p90 = float(np.percentile(vals, 90))
        bmd_value = self._bmd_score(mean_d, dens, vals)

        # Overlay: draw the actual segmentation polygon outline for every
        # detected vertebra. L4 (target) in red + light tint, others in blue.
        vis = rgb.copy()
        # Fill L4 ROI with a light translucent red tint (keep bone visible)
        red_tint = vis.copy()
        red_tint[mask > 0] = (
            0.75 * red_tint[mask > 0] + 0.25 * np.array([255, 0, 0])
        ).astype(np.uint8)
        vis = red_tint
        # Draw thin polygon outlines, with labels placed OUTSIDE (above-left)
        for i in range(len(boxes)):
            poly = np.asarray(r.masks.xy[i], dtype=np.int32)
            if poly.size == 0:
                continue
            is_l4 = i == l4_i
            color = (255, 0, 0) if is_l4 else (0, 170, 255)  # RGB
            thick = 2 if is_l4 else 1
            cv2.polylines(vis, [poly], isClosed=True, color=color, thickness=thick)
            # Place the label OUTSIDE the polygon, vertically centered, to
            # the LEFT of the box (or to the right if there's no room left).
            # This avoids overlapping the vertebra above/below.
            cid = classes[i]
            label = CLASS_NAMES[cid] if 0 <= cid < len(CLASS_NAMES) else str(cid)
            lx = int(poly[:, 0].min())
            rx = int(poly[:, 0].max())
            cy_box = int((poly[:, 1].min() + poly[:, 1].max()) / 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            fscale = 0.55
            fthick = 2 if is_l4 else 1
            (tw, th), _ = cv2.getTextSize(label, font, fscale, fthick)
            gap = 8
            if lx - gap - tw >= 0:
                tx = lx - gap - tw          # left of the box
            else:
                tx = min(rx + gap, W - tw)  # fallback: right of the box
            ty = cy_box + th // 2           # vertically centered
            cv2.putText(
                vis, label, (tx, ty),
                font, fscale, color, fthick, cv2.LINE_AA,
            )
        overlay_name = "l4_overlay.png"
        cv2.imwrite(
            str(output_dir / overlay_name),
            cv2.cvtColor(vis, cv2.COLOR_RGB2BGR),
        )

        # Extract a zoomed-in crop of the L4 vertebra for prominent display.
        # IMPORTANT: crop from the CLEAN original image (rgb), NOT the
        # polygon-annotated 'vis', so the crop shows the actual bone with no
        # outlines or tint obscuring it.
        l4_crop_name = None
        x1, y1, x2, y2 = boxes[l4_i].astype(int)
        pad_x = int((x2 - x1) * 0.25)
        pad_y = int((y2 - y1) * 0.25)
        cx1 = max(x1 - pad_x, 0)
        cy1 = max(y1 - pad_y, 0)
        cx2 = min(x2 + pad_x, W)
        cy2 = min(y2 + pad_y, H)
        if cx2 > cx1 and cy2 > cy1:
            crop = rgb[cy1:cy2, cx1:cx2]  # clean original, no annotations
            # Upscale so the L4 detail is large and clear (min width 480px)
            ch, cw = crop.shape[:2]
            if cw < 480 and cw > 0:
                scale = 480 / cw
                crop = cv2.resize(
                    crop,
                    (int(cw * scale), int(ch * scale)),
                    interpolation=cv2.INTER_CUBIC,
                )
            l4_crop_name = "l4_crop.png"
            cv2.imwrite(
                str(output_dir / l4_crop_name),
                cv2.cvtColor(crop, cv2.COLOR_RGB2BGR),
            )

        # L4 마스크 npy 저장 (재현/감사용)
        np.save(str(output_dir / "l4_mask.npy"), mask)

        # XAI contribution items (statistics-based interpretation)
        xai = self._build_xai(
            bmd_value, mean_d, median_d, std_d, p10, p90, float(confs[l4_i])
        )

        # 신뢰도 = L4 검출 confidence
        confidence = round(float(confs[l4_i]), 3)

        # Parse acquisition date (YYYYMMDD) + time (HHMMSS) if present
        acquired_at = None
        if acquired and len(str(acquired)) == 8:
            s = str(acquired)
            tm = "00:00:00"
            if acquired_time:
                t = str(acquired_time).split(".")[0]  # drop fractional secs
                if len(t) >= 6:
                    tm = f"{t[:2]}:{t[2:4]}:{t[4:6]}"
            acquired_at = f"{s[:4]}-{s[4:6]}-{s[6:8]}T{tm}"

        l4_crop_path = (
            f"{output_dir.name}/{l4_crop_name}" if l4_crop_name else None
        )

        return InferenceResult(
            bmd_value=bmd_value,
            target_vertebra="L4",
            t_score=None,  # 참조군 분포가 있으면 산출. 현재 미제공.
            confidence=confidence,
            exposure_corrected=True,
            model_version=self.MODEL_VERSION,
            preview_path=f"{output_dir.name}/{preview_name}",
            gradcam_path=f"{output_dir.name}/{overlay_name}",
            l4_crop_path=l4_crop_path,
            segments=segments,
            xai_factors=xai,
            acquired_at=acquired_at,
            modality=modality,
            dicom_meta=dicom_meta,
        )

    @staticmethod
    def _build_xai(
        bmd_value, mean_d, median_d, std_d, p10, p90, conf
    ) -> list[XaiResult]:
        """Build clinician-readable XAI contribution items from L4 ROI stats.

        Translates summary statistics into human-readable rationale.
        `contribution` is a -1..+1 relative weight (interpretation aid only).
        """
        spread = std_d / (abs(mean_d) + 1e-6)  # coefficient of variation
        skew = (mean_d - median_d) / (abs(median_d) + 1e-6)
        dynamic = (p90 - p10) / (abs(mean_d) + 1e-6)

        factors = [
            XaiResult(
                label="L4 ROI mean attenuation",
                # Use the per-image normalized BMD so this stays in 0..1
                contribution=round(float(np.clip(bmd_value, -1, 1)), 3),
                description=(
                    f"Normalized density {bmd_value:.3f} "
                    f"(raw mean {mean_d:.0f})"
                ),
            ),
            XaiResult(
                label="Density uniformity",
                contribution=round(float(np.clip(-spread, -1, 1)), 3),
                description=f"Coefficient of variation {spread:.3f} (lower = more uniform)",
            ),
            XaiResult(
                label="Intensity dynamic range",
                contribution=round(float(np.clip(-dynamic * 0.5, -1, 1)), 3),
                description=f"p10–p90 spread {p90 - p10:.1f}",
            ),
            XaiResult(
                label="Distribution skew",
                contribution=round(float(np.clip(skew, -1, 1)), 3),
                description=f"Mean–median offset {skew:+.3f}",
            ),
            XaiResult(
                label="L4 detection confidence",
                contribution=round(float(np.clip(conf, -1, 1)), 3),
                description=f"Segmentation model confidence {conf:.2f}",
            ),
        ]
        # Sort by absolute contribution magnitude
        factors.sort(key=lambda f: abs(f.contribution), reverse=True)
        return factors
