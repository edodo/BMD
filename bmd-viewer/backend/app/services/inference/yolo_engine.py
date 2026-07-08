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
    PartialInferenceError,
    SegmentResult,
    XaiResult,
)

logger = logging.getLogger(__name__)

CLASS_NAMES = ["L1", "L2", "L3", "L4", "L5"]
L4_CLASS_ID = CLASS_NAMES.index("L4")  # 3

# 노트북 하이퍼파라미터 (L4_AP_segmentation_v9.ipynb — YOLO26m-seg, AP+LA)
INVERT = True          # MONOCHROME1을 학습 톤에 맞춰 뼈를 밝게
CONF = 0.25            # detection confidence threshold
IMGSZ = 640            # 학습/ONNX export 입력 크기 (dynamic=False로 고정)
# 해면골 ROI 크기(ROI_SHRINK)는 settings.bmd_roi_shrink로 이동 — 런타임 튜닝 가능.
# 1.0=L4 전체, 0.85=L4에 딱 맞음(피질골 edge 회피, 모델 소스와 동일), 0.7=좁은 중심.

# BMD is normalized per-image (relative to this image's own soft-tissue
# baseline and bone peak), so no fixed reference constant is used.
# This keeps the score stable across different DICOM bit depths/exposures.

# XAI Eigen-CAM 히트맵 합성 비율 (0.55*원본 + 0.45*히트맵), 노트북 v3 동일
CAM_BLEND = 0.45


class _YoloEigenCAM:
    """Gradient-free Eigen-CAM (노트북 v3 이식).

    YOLO 네크(neck)의 고해상도 활성맵을 forward hook으로 포착한 뒤,
    그 주성분(PCA via SVD)을 클래스-무관 saliency 맵으로 사용한다.
    deep-bottleneck Grad-CAM과 달리 L4 본체/골소주에 어트리뷰션이 위치한다.

    훅은 모델당 1회만 등록하고, predict 호출마다 caps를 비워 재사용한다.
    """

    def __init__(self, yolo_model) -> None:
        self.caps: dict[int, object] = {}
        self._handles: list = []
        layers = yolo_model.model.model  # nn.Sequential of YOLO layers
        # 네크(마지막 1/3 구간): 의미 정보가 있으면서 공간 해상도가 남아있는 층
        for idx in range(max(0, len(layers) - 12), len(layers)):
            self._handles.append(
                layers[idx].register_forward_hook(self._mk(idx))
            )

    def _mk(self, idx: int):
        def hook(_m, _i, out):
            t = out[0] if isinstance(out, (list, tuple)) else out
            if getattr(t, "ndim", 0) == 4 and t.shape[1] >= 32:
                self.caps[idx] = t.detach()
        return hook

    def clear(self) -> None:
        self.caps = {}

    def compute(self) -> "np.ndarray | None":
        """포착된 활성맵에서 0~1 정규화된 CAM(h, w)을 산출."""
        if not self.caps:
            return None
        # 공간 해상도가 가장 큰(가장 세밀한) 네크 출력을 선택
        idx = max(self.caps, key=lambda k: self.caps[k].shape[2] * self.caps[k].shape[3])
        A = self.caps[idx][0].cpu().numpy()       # (C, h, w)
        C, h, w = A.shape
        X = A.reshape(C, h * w).T                  # (hw, C)
        X = X - X.mean(0, keepdims=True)
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        cam = (X @ Vt[0]).reshape(h, w)           # 제1주성분
        if cam.mean() < 0:                        # 강한 활성 방향으로 정렬
            cam = -cam
        cam = np.maximum(cam, 0)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)
        return cam


class YoloBmdEngine(BmdInferenceEngine):
    """YOLOv8-seg 기반 L1~L5 분할 + L4 proxy BMD 산출 엔진."""

    MODEL_VERSION = "yolo26m-seg-l1l5-ap-la-2.0"

    def __init__(self, weights_path: Path | None = None) -> None:
        self._weights = weights_path or (
            settings.ml_model_dir / settings.ml_model_file
        )
        self._model = None  # lazy load
        self._cam = None    # lazy Eigen-CAM hooker

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

    def _ensure_cam(self):
        """Eigen-CAM 훅을 모델당 1회 등록 (실패 시 None).

        ONNX(onnxruntime) 백엔드에는 torch nn.Module 층이 없어 forward hook을
        걸 수 없다. 이 경우 Eigen-CAM(모델 주목도 크롭)은 생략하고 밀도 근거
        히트맵(xai_density.png)만 산출한다.
        """
        if self._weights.suffix.lower() == ".onnx":
            self._cam = False  # ONNX: torch hook 불가 → CAM 생략
            return None
        if self._cam is None:
            try:
                self._cam = _YoloEigenCAM(self._ensure_model())
            except Exception as exc:  # noqa: BLE001
                logger.warning("Eigen-CAM 초기화 실패 (XAI 히트맵 생략): %s", exc)
                self._cam = False  # 재시도 방지용 sentinel
        return self._cam or None

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
    def _crop_box(H, W, xyxy, pad=0.25):
        """L4 bbox + 여유 패딩을 영상 경계로 클립한 (cx1, cy1, cx2, cy2)."""
        x1, y1, x2, y2 = (int(v) for v in xyxy)
        px = int((x2 - x1) * pad)
        py = int((y2 - y1) * pad)
        return (
            max(x1 - px, 0),
            max(y1 - py, 0),
            min(x2 + px, W),
            min(y2 + py, H),
        )

    @staticmethod
    def _upscale(img, min_w=480):
        """좁은 패널에서도 또렷하게 보이도록 최소 너비까지 확대."""
        h, w = img.shape[:2]
        if 0 < w < min_w:
            s = min_w / w
            img = cv2.resize(
                img, (int(w * s), int(h * s)), interpolation=cv2.INTER_CUBIC
            )
        return img

    @classmethod
    def _build_density_heatmap(cls, rgb, dens, mask, low, high, box):
        """충실한 BMD 근거: L4 ROI를 '실제 픽셀 감쇠'로 색칠한 L4 크롭.

        각 픽셀 점수 = clip((dens-low)/(high-low), 0, 1) — BMD 점수의 픽셀
        단위 버전이다. 따뜻한 색=고밀도(BMD↑), 차가운 색=저밀도. 색은 ROI
        내부에만 입히고 주변 L4 본체는 회색조로 남겨 위치 맥락을 보존한다.
        신경망 feature가 아니라 'BMD 숫자를 만든 픽셀' 그 자체를 보여준다.
        """
        score = np.clip((dens - low) / max(high - low, 1e-6), 0, 1)
        heat = cv2.cvtColor(
            cv2.applyColorMap((score * 255).astype(np.uint8), cv2.COLORMAP_JET),
            cv2.COLOR_BGR2RGB,
        )
        base = rgb.copy()
        m = mask > 0
        base[m] = (0.30 * base[m] + 0.70 * heat[m]).astype(np.uint8)
        # ROI 외곽선(초록 테두리)은 표시하지 않는다 — 히트맵만 깔끔하게.
        cx1, cy1, cx2, cy2 = box
        return cls._upscale(base[cy1:cy2, cx1:cx2])

    @classmethod
    def _build_l4_cam_crop(cls, rgb, cam, box):
        """분할 모델 주목도: L4 크롭에 Eigen-CAM을 입힌 어트리뷰션(노트북 v3).

        cam(저해상도)을 원본 크기로 확대 후 L4 크롭 영역만 잘라 정규화 →
        JET 컬러맵 → 깨끗한 크롭과 블렌드. 크롭으로 한정하므로 히트맵이
        항상 L4 위에 있고 내부 골소주(trabecular) 패턴 주목까지 드러난다.
        """
        H, W = rgb.shape[:2]
        cam_full = cv2.resize(cam, (W, H))
        cx1, cy1, cx2, cy2 = box
        crop_rgb = rgb[cy1:cy2, cx1:cx2]
        crop_cam = cam_full[cy1:cy2, cx1:cx2]
        crop_cam = (crop_cam - crop_cam.min()) / (
            crop_cam.max() - crop_cam.min() + 1e-6
        )
        crop_heat = cv2.cvtColor(
            cv2.applyColorMap((crop_cam * 255).astype(np.uint8), cv2.COLORMAP_JET),
            cv2.COLOR_BGR2RGB,
        )
        blend = (
            (1 - CAM_BLEND) * crop_rgb + CAM_BLEND * crop_heat
        ).astype(np.uint8)
        return cls._upscale(blend)

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
    def _reliability(
        l4_box, H: int, W: int, n_vertebrae: int, margin_frac: float = 0.02
    ) -> tuple[bool, str | None]:
        """측정 신뢰도 점검. 값은 유지하되 부적합 신호를 경고로 반환한다.

        주 신호는 '경계 잘림(truncation)': L4 bbox가 이미지 상/하/좌/우 경계에
        여백(margin_frac) 이내로 닿으면, 실제 L4가 프레임 밖으로 잘려나간
        (예: 골반/고관절 영상) 것이므로 ROI가 불완전 → BMD 신뢰 불가.
        보조로 검출 척추 수가 너무 적으면 요추 영상이 아닐 가능성을 덧붙인다.
        """
        x1, y1, x2, y2 = l4_box
        m = max(2.0, margin_frac * min(H, W))
        sides = []
        if y1 <= m:
            sides.append("top")
        if y2 >= H - m:
            sides.append("bottom")
        if x1 <= m:
            sides.append("left")
        if x2 >= W - m:
            sides.append("right")

        reasons = []
        if sides:
            reasons.append(
                "L4 vertebra touches the image edge ("
                + ", ".join(sides)
                + ") — likely cut off (e.g. pelvis/hip view); "
                "the ROI is partial so BMD is unreliable."
            )
        if n_vertebrae < 3:
            reasons.append(
                f"Only {n_vertebrae} vertebra(e) detected — may not be a "
                "lumbar-spine view suitable for L4 BMD."
            )
        if reasons:
            return False, " ".join(reasons)
        return True, None

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
    def _valid_tissue_mask(dens: np.ndarray) -> np.ndarray:
        """유효 조직(연부조직+뼈) 마스크. 배경(공기)과 금속 인공물을 제외한다.

        고관절 수술 환자 영상 대응:
          - 배경 : Otsu 임계 이하 = 거의 균일한 저밀도 영역(공기/촬영 여백).
                   전체 percentile 기준선을 아래로 끌어내려 BMD를 부풀리므로 제외.
          - 금속 : 유효 조직 p95(치밀골 근방) × metal_ratio 초과 = 초고감쇠 인공물
                   (인공관절/척추 고정재). 상단 percentile을 끌어올리므로 제외.
        금속이 없는 영상에서는 max가 임계를 넘지 않아 아무 것도 제외되지 않는다.
        """
        valid = np.ones(dens.shape, dtype=bool)
        if settings.bmd_exclude_background:
            u8 = cv2.normalize(dens, None, 0, 255, cv2.NORM_MINMAX).astype(
                np.uint8
            )
            thr, _ = cv2.threshold(
                u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            valid &= u8 >= thr  # 배경/공기 제외 → 몸통(전경)만
        body = dens[valid]
        if body.size:
            # 금속에 덜 휘둘리는 뼈 기준값(p95)의 배수로 금속 임계 설정
            bone_ref = float(np.percentile(body, 95))
            metal_thr = settings.bmd_metal_ratio * bone_ref
            valid &= dens < metal_thr  # 금속/인공물 제외
        return valid

    @staticmethod
    def _density_scale(
        dens: np.ndarray,
        valid: np.ndarray,
        l4_box: tuple[int, int, int, int],
        bone_union: np.ndarray,
        roi_vals: np.ndarray,
    ) -> tuple[float, float]:
        """BMD 정규화용 low/high 기준값 (유효 조직 기반).

          - low  = L4 인접 '국소 연부조직' 중앙값 (뼈 마스크 제외).
                   측정 부위 바로 옆 조직을 기준 삼으므로 촬영 시야/배경에 면역.
          - high = 유효 조직의 99 percentile (치밀골). 금속은 이미 제외됨.
        국소 연부조직 픽셀이 부족하면 유효 조직 전체 25 percentile로 폴백,
        그래도 축퇴(degenerate)면 ROI 자체 5–95 percentile을 쓴다.
        BMD 점수와 밀도 히트맵이 동일 스케일을 공유하도록 한 곳에서 산출.
        """
        cx1, cy1, cx2, cy2 = l4_box
        window = np.zeros(dens.shape, dtype=bool)
        window[cy1:cy2, cx1:cx2] = True
        soft = valid & window & (bone_union == 0)
        soft_vals = dens[soft]
        if soft_vals.size >= 50:
            low = float(np.median(soft_vals))
        else:
            body = dens[valid]
            low = float(np.percentile(body, 25)) if body.size else float(
                np.percentile(dens, 25)
            )

        body = dens[valid]
        high = float(np.percentile(body, 99)) if body.size else float(
            np.percentile(dens, 99)
        )
        if high - low <= 1e-6:
            roi_lo = float(np.percentile(roi_vals, 5))
            roi_hi = float(np.percentile(roi_vals, 95))
            return roi_lo, roi_lo + max(roi_hi - roi_lo, 1e-6)
        return low, high

    @staticmethod
    def _bmd_score(mean_dens: float, low: float, high: float) -> float:
        """L4 ROI 평균 감쇠를 [low, high] → 0..1 proxy BMD 점수로 사상."""
        score = (mean_dens - low) / (high - low)
        return round(float(np.clip(score, 0.0, 1.0)), 3)

    @classmethod
    def _draw_detection_overlay(cls, rgb, masks_xy, classes, boxes, W, H, l4_i=None):
        """검출된 척추들의 폴리곤 외곽선 + 라벨을 그린 오버레이.

        부분 실패(L4 식별 불가) 시 '추출한 것'을 화면에 보여주기 위해 쓰인다.
        l4_i가 주어지면 해당 척추만 빨강으로 강조, 그 외는 파랑.
        """
        vis = rgb.copy()
        for i in range(len(boxes)):
            poly = np.asarray(masks_xy[i], dtype=np.int32)
            if poly.size == 0:
                continue
            is_l4 = l4_i is not None and i == l4_i
            color = (255, 0, 0) if is_l4 else (0, 170, 255)
            thick = 2 if is_l4 else 1
            cv2.polylines(vis, [poly], isClosed=True, color=color, thickness=thick)
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
            tx = lx - gap - tw if lx - gap - tw >= 0 else min(rx + gap, W - tw)
            ty = cy_box + th // 2
            cv2.putText(
                vis, label, (tx, ty), font, fscale, color, fthick, cv2.LINE_AA
            )
        return vis

    @staticmethod
    def _density_grid(dens, mask, valid, low, high, n):
        """L4 ROI를 bbox 기준 N×N 격자로 리샘플한 정규화 밀도(0..1).

        종적 대조(compare)용. 서로 다른 두 X-ray의 L4를 각자 bbox로 정규화하므로
        셀(i,j)은 두 검사에서 '척추체 내 같은 상대 위치'를 뜻한다(강체정합은 아님).
        post − pre 격자가 음수인 셀 = 그 부위 골밀도 감소(골손실).
        각 셀 값 = (mask & valid) 픽셀의 평균 score, 유효 픽셀 없으면 None.
        """
        ys, xs = np.where(mask > 0)
        if ys.size == 0:
            return None
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        score = np.clip((dens - low) / max(high - low, 1e-6), 0.0, 1.0)
        sel = (mask > 0) & valid
        y_edges = np.linspace(y0, y1, n + 1).astype(int)
        x_edges = np.linspace(x0, x1, n + 1).astype(int)
        grid: list[list[float | None]] = []
        for r in range(n):
            row: list[float | None] = []
            for c in range(n):
                cy0, cy1 = y_edges[r], y_edges[r + 1]
                cx0, cx1 = x_edges[c], x_edges[c + 1]
                cell = sel[cy0:cy1, cx0:cx1]
                if cell.any():
                    vals = score[cy0:cy1, cx0:cx1][cell]
                    row.append(round(float(vals.mean()), 4))
                else:
                    row.append(None)
            grid.append(row)
        return grid

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
        preview_rel = f"{output_dir.name}/{preview_name}"

        # Eigen-CAM 훅 준비: predict 동안 네크 활성맵을 포착한다.
        cam_hooker = self._ensure_cam()
        if cam_hooker is not None:
            cam_hooker.clear()

        r = model.predict(rgb, conf=CONF, imgsz=IMGSZ, verbose=False)[0]
        if r.masks is None or len(r.boxes) == 0:
            # 원본은 이미 저장됨 → 화면에 원본만 보여주고 실패 표시
            raise PartialInferenceError(
                "No vertebrae detected (check image quality/exposure)",
                preview_path=preview_rel,
            )

        boxes = r.boxes.xyxy.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()

        l4_i, status = self._pick_l4_index(boxes, classes, confs)
        if l4_i is None:
            # L4는 못 찾았지만 검출된 척추는 그려서 '추출한 것'까지 보여준다
            det_vis = self._draw_detection_overlay(
                rgb, r.masks.xy, classes, boxes, W, H
            )
            det_name = "detection.png"
            cv2.imwrite(
                str(output_dir / det_name),
                cv2.cvtColor(det_vis, cv2.COLOR_RGB2BGR),
            )
            raise PartialInferenceError(
                f"Could not identify L4: {status}",
                preview_path=preview_rel,
                overlay_path=f"{output_dir.name}/{det_name}",
            )

        # 측정 신뢰도: 대상 척추가 이미지 경계에 잘렸는지 등 (값은 유지, 경고만).
        reliable, reliability_warning = self._reliability(
            boxes[l4_i], H, W, len(boxes)
        )
        if not reliable:
            logger.warning("측정 신뢰도 낮음: %s", reliability_warning)

        # 유효 조직 마스크(배경·금속 제외)와 전체 뼈 영역(연부조직 기준선 산출용).
        valid = self._valid_tissue_mask(dens)
        bone_union = np.zeros((H, W), np.uint8)

        # 검출된 모든 척추를 분할 결과로 기록
        segments: list[SegmentResult] = []
        for i in range(len(boxes)):
            cid = classes[i]
            label = CLASS_NAMES[cid] if 0 <= cid < len(CLASS_NAMES) else str(cid)
            m_full = self._poly_to_mask(r.masks.xy[i], H, W, shrink=1.0)
            bone_union |= m_full
            seg_vals = dens[(m_full > 0) & valid]
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

        # L4 ROI (중앙 해면골) 골밀도 통계. 유효 조직만 사용(ROI 내 금속 픽셀 배제).
        mask = self._poly_to_mask(
            r.masks.xy[l4_i], H, W, shrink=settings.bmd_roi_shrink
        )
        vals = dens[(mask > 0) & valid]
        if vals.size == 0:
            # 전부 금속/배경으로 걸러진 극단 상황 → 원 ROI로 폴백
            vals = dens[mask > 0]
        if vals.size == 0:
            det_vis = self._draw_detection_overlay(
                rgb, r.masks.xy, classes, boxes, W, H, l4_i=l4_i
            )
            det_name = "detection.png"
            cv2.imwrite(
                str(output_dir / det_name),
                cv2.cvtColor(det_vis, cv2.COLOR_RGB2BGR),
            )
            raise PartialInferenceError(
                "L4 detected but ROI is empty (mask/exposure issue)",
                preview_path=preview_rel,
                overlay_path=f"{output_dir.name}/{det_name}",
            )

        mean_d = float(vals.mean())
        median_d = float(np.median(vals))
        std_d = float(vals.std())
        p10 = float(np.percentile(vals, 10))
        p90 = float(np.percentile(vals, 90))
        # low = L4 인접 국소 연부조직, high = 유효 조직 99pct (배경·금속 제외).
        # 뼈 경계가 연부조직에 새지 않도록 뼈 영역을 살짝 팽창시켜 사용.
        bone_dil = cv2.dilate(bone_union, np.ones((7, 7), np.uint8), 1)
        low_anchor, high_anchor = self._density_scale(
            dens, valid, self._crop_box(H, W, boxes[l4_i], pad=0.6), bone_dil,
            vals,
        )
        bmd_value = self._bmd_score(mean_d, low_anchor, high_anchor)
        # 종적 대조용 L4 밀도 격자 (post−pre로 골손실 부위 검출)
        density_grid = self._density_grid(
            dens, mask, valid, low_anchor, high_anchor, settings.bmd_diff_grid_n
        )

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

        # XAI 시각화 두 가지 (각각 격리: 실패해도 본체 추론엔 영향 없음).
        # 둘 다 L4 bbox로 크롭하므로 히트맵이 항상 L4 위에 위치한다.
        #   1) 밀도 근거: L4 ROI를 '실제 감쇠'로 색칠 → BMD 숫자의 직접 근거
        #   2) 모델 주목도: L4 크롭 Eigen-CAM 어트리뷰션
        xai_overlay_name = None   # 밀도 근거 히트맵 (faithful)
        xai_l4_cam_name = None    # Eigen-CAM 크롭
        l4_box = self._crop_box(H, W, boxes[l4_i])
        try:
            dens_heat = self._build_density_heatmap(
                rgb, dens, mask, low_anchor, high_anchor, l4_box
            )
            xai_overlay_name = "xai_density.png"
            cv2.imwrite(
                str(output_dir / xai_overlay_name),
                cv2.cvtColor(dens_heat, cv2.COLOR_RGB2BGR),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("밀도 히트맵 생성 실패 (생략): %s", exc)
            xai_overlay_name = None
        if cam_hooker is not None:
            try:
                cam = cam_hooker.compute()
                if cam is not None:
                    cam_crop = self._build_l4_cam_crop(rgb, cam, l4_box)
                    xai_l4_cam_name = "xai_l4_cam.png"
                    cv2.imwrite(
                        str(output_dir / xai_l4_cam_name),
                        cv2.cvtColor(cam_crop, cv2.COLOR_RGB2BGR),
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Eigen-CAM 크롭 생성 실패 (생략): %s", exc)
                xai_l4_cam_name = None

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
        xai_overlay_path = (
            f"{output_dir.name}/{xai_overlay_name}" if xai_overlay_name else None
        )
        xai_l4_cam_path = (
            f"{output_dir.name}/{xai_l4_cam_name}" if xai_l4_cam_name else None
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
            xai_overlay_path=xai_overlay_path,
            xai_l4_cam_path=xai_l4_cam_path,
            # 밀도 히트맵 스케일 (파랑 끝 / 빨강 끝 / L4 평균)
            density_low=round(low_anchor, 1),
            density_high=round(high_anchor, 1),
            roi_mean_attenuation=round(mean_d, 1),
            l4_density_grid=density_grid,
            reliable=reliable,
            reliability_warning=reliability_warning,
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
