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
import math
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

# XAI Seg-Grad-CAM 히트맵 합성 비율 (0.55*원본 + 0.45*히트맵), 노트북 v18 E1 동일
CAM_BLEND = 0.45

# ---------- Precision / LSC (v18 노트북 D0와 동일 산식) ----------
# 재촬영 사이 재위치(repositioning) 노이즈를 흉내내는 강체+스케일 섭동.
# 노트북 v18의 LSC_JITTER와 동일한 값 — 두 시각(visit) 사이 실제 변화가 아닌
# '같은 환자를 다시 찍었을 때의 위치 오차'만큼만 흔든다.
JITTER_DEG = 2.0          # +/- 회전 (도)
JITTER_SHIFT_FRAC = 0.01  # +/- 이동 (이미지 크기 대비 비율)
JITTER_SCALE = 0.02       # +/- 배율
LSC_N_REPEATS = 5         # 스터디당 재측정 횟수
LSC_Z95 = 2.77            # ISCD 95% 신뢰수준 계수 (RMS-CV% -> LSC%)

# ---------- 텍스처(골소주 미세구조) + AE 이상점수 (노트북 v18 Sec.C/B4) ----------
GLCM_LEVELS = 32       # GLCM 그레이레벨 양자화 단계
VARIO_MAX_LAG = 8      # variogram(TBS 유사) 최대 lag(px)
FRACTAL_BLOCK = 31     # box-counting 전 적응적 이진화 커널 크기
AE_CHECKPOINT_FILE = "spine_ae_best.pt"
AE_SCALER_FILE = "feature_scaler_mean_std.json"
AE_REFERENCE_FILE = "ae_reference_features.csv"


# Seg-Grad-CAM 네크 레이어 (노트북 v18 E1). 실측 확정값 — v17이 물려준 인덱스
# 15("P3 neck feature"라던 주석)는 이 모델의 실제 층 그래프에서 학습 파라미터가
# 없는 bare Concat이라 그래디언트가 의미 없었다(off-by-one, 가정이 아니라 실측
# 으로 확인). 층 그래프를 직접 나열해([type(m).__name__ for m in net.model])
# Concat을 처리하는 다음 C3k2인 16으로 수정 — 4장 스윕에서 energy-in-spine
# 62.3%/pointing-game 100% (레이어 15는 13.0%/50%).
CAM_NECK_LAYER = 16
CAM_PROTO_SCALE = 4  # prototype-mask 해상도 = IMGSZ // 4


class _SegGradCAM:
    """실제 그래디언트 기반 Seg-Grad-CAM (노트북 v18 E1 이식 — Eigen-CAM 대체).

    Eigen-CAM(클래스 무관, PCA 기반, v3~v2 세대)은 네크 활성맵의 제1주성분을
    쓰기 때문에 영상 내 모든 뼈(골반·늑골·대퇴골두 포함)에 반응했다 —
    "모델이 왜 L4를 그렇게 골랐는가"에 대한 답이 아니었다. 이 구현은 검출된
    L1~L5 '인스턴스 마스크 점수'의 합에 대해 실제 backward 1회를 수행해,
    척추 마스크 형성에 실제로 기여한 활성만 강조한다.

    별도 오토그래드 전용 모델 인스턴스가 필요하다: 파이프라인의 주 추론
    모델은 이미 여러 번 predict()를 거쳐 앵커 버퍼가 추론 전용(non-autograd)
    텐서로 캐시돼 있어 그래프에 들어갈 수 없다(신선한 로드는 이 문제가 없다).
    """

    def __init__(self, weights_path: Path) -> None:
        import torch
        from ultralytics import YOLO

        net = YOLO(str(weights_path)).model.eval()
        for p in net.parameters():
            p.requires_grad_(True)
        self._net = net
        self._device = next(net.parameters()).device
        self._torch = torch

    def compute(
        self, img_bgr: np.ndarray, layer_idx: int = CAM_NECK_LAYER
    ) -> tuple["np.ndarray | None", dict]:
        """(cam[H,W] in [0,1], {cls_id: bool mask[H,W]}).

        검출이 없거나(L1~L5 미검출) 지정 레이어가 마스크 형성의 그래프
        상류가 아니면(grad=None) (None, {})."""
        from ultralytics.data.augment import LetterBox

        torch = self._torch
        H0, W0 = img_bgr.shape[:2]
        r = min(IMGSZ / H0, IMGSZ / W0)
        nw, nh = round(W0 * r), round(H0 * r)
        pw, ph = (IMGSZ - nw) // 2, (IMGSZ - nh) // 2
        im = LetterBox((IMGSZ, IMGSZ), auto=False)(image=img_bgr)
        t = (
            torch.from_numpy(im[:, :, ::-1].transpose(2, 0, 1).copy())
            .float()
            .unsqueeze(0)
            .to(self._device)
            / 255.0
        )

        def _unletter(x, interp):
            x = cv2.resize(x, (IMGSZ, IMGSZ), interpolation=interp)[
                ph : ph + nh, pw : pw + nw
            ]
            return cv2.resize(x, (W0, H0), interpolation=interp)

        t = t.requires_grad_(True)
        acts: dict = {}
        layer = self._net.model[layer_idx]
        handle = layer.register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))
        masks_np: dict[int, np.ndarray] = {}
        proto_res = IMGSZ // CAM_PROTO_SCALE
        try:
            self._net.model[-1].shape = None  # 오토그래드 하에서 앵커 재계산 강제
            with torch.enable_grad():
                det, proto = self._net(t)[0]
                det, proto = det[0], proto[0]
                K = proto.shape[0]
                score_v, clsid = det[:, 4], det[:, 5]
                total: "torch.Tensor | float" = 0.0
                for c in range(len(CLASS_NAMES)):
                    selc = torch.where((clsid.round() == c) & (score_v > 0))[0]
                    if not len(selc):
                        continue
                    a = selc[torch.argmax(score_v[selc])]
                    coeff = det[a][6 : 6 + K]
                    prob = torch.sigmoid((coeff[:, None, None] * proto).sum(0))
                    crop = torch.zeros_like(prob)
                    x0, y0, x1, y1 = [
                        int(v / CAM_PROTO_SCALE) for v in det[a][:4]
                    ]
                    x0, y0 = max(0, x0), max(0, y0)
                    x1, y1 = min(proto_res, x1), min(proto_res, y1)
                    crop[y0:y1, x0:x1] = 1.0
                    prob_c = prob * crop
                    total = total + (prob_c * (prob_c > 0.5).float().detach()).sum()
                    masks_np[c] = prob_c.detach().cpu().numpy().astype(np.float32)
                if not isinstance(total, torch.Tensor):
                    return None, {}
                A = acts.get("a")
                if A is None:
                    return None, {}
                g = torch.autograd.grad(
                    total, A, retain_graph=False, allow_unused=True
                )[0]
                if g is None:
                    logger.warning(
                        "Grad-CAM layer %d is not upstream of the masks "
                        "(grad is None)", layer_idx,
                    )
                    return None, {}
                Aw, gw = A[0].detach(), g[0].detach()
                cam = (
                    torch.relu((gw.mean(dim=(1, 2), keepdim=True) * Aw).sum(0))
                    .cpu()
                    .numpy()
                )
        finally:
            handle.remove()

        cam = _unletter(cam.astype(np.float32), cv2.INTER_LINEAR)
        cam = cv2.GaussianBlur(cam, (0, 0), sigmaX=max(H0, W0) / 150.0)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)
        masks = {c: _unletter(m, cv2.INTER_LINEAR) > 0.5 for c, m in masks_np.items()}
        return cam, masks


def _build_spine_feature_ae_class():
    """torch.nn.Module 서브클래스는 지연 정의 — 이 파일의 다른 모델(YOLO)과
    동일하게 무거운 프레임워크 import를 실제 사용 시점까지 미룬다."""
    import torch.nn as nn

    class SpineFeatureAE(nn.Module):
        """표준화된 척추별 특징벡터의 대칭 AE (노트북 v18 Sec.C2, 추론 전용).

        encoder: input_dim -> hidden[0] -> ... -> hidden[-1] -> latent_dim
        decoder: latent_dim -> hidden[-1] -> ... -> hidden[0] -> input_dim
        (대칭, 출력은 선형 — 표준화된 특징은 확률/유계값이 아니므로 마지막에
        활성함수를 씌우면 꼬리값(이상점수가 가장 신경쓰는 극단값) 복원이 편향된다.)
        """

        def __init__(self, input_dim, hidden, latent_dim, dropout, activation):
            super().__init__()
            enc, prev = [], input_dim
            for h in hidden:
                enc += [nn.Linear(prev, h), activation(), nn.Dropout(dropout)]
                prev = h
            enc += [nn.Linear(prev, latent_dim)]
            self.encoder = nn.Sequential(*enc)

            dec, prev = [], latent_dim
            for h in reversed(hidden):
                dec += [nn.Linear(prev, h), activation(), nn.Dropout(dropout)]
                prev = h
            dec += [nn.Linear(prev, input_dim)]
            self.decoder = nn.Sequential(*dec)

        def forward(self, x):
            z = self.encoder(x)
            return self.decoder(z), z

    return SpineFeatureAE


class _AnomalyScorer:
    """AE 재구성오차 기반 이상점수 (노트북 v18 Sec.C9, v18_spec.md Sec.5).

    척추별 특징벡터(BAR + GLCM 5종 + fractal + variogram, 표준화)를 학습된
    오토인코더로 복원하고, 복원오차(MSE)를 '정상'(PASS-QC) 코호트의 오차
    분포 대비 백분위로 환산한다. 이 배포는 재학습하지 않는다 — 체크포인트는
    노트북에서 학습된 것을 그대로 추론 전용으로 불러온다(세그멘테이션 모델과
    동일 원칙). anomaly_pct는 보정된 확률이 아니라 '이 코호트 내에서 이
    복원오차가 몇 번째 백분위인가'일 뿐이며, 라벨이 없으므로 임상적 확정
    진단이 아니다(v18_spec.md Sec.4 정직성 원칙과 동일).

    기준(reference) 분포는 노트북이 실제로 학습에 쓴 held-out validation
    오차가 아니라, 학습에 쓰인 PASS-QC 코호트(ae_reference_features.csv,
    노트북의 df_fit) 전체에서 이 로드 시점에 재계산한 오차다 — 정확한 val
    분할은 저장되지 않았으므로 가장 가까운 근사치이며, '정상' 코호트의
    분포를 앵커로 쓴다는 취지는 그대로 유지된다.
    """

    FEATURE_COLS = [
        "bar", "glcm_contrast", "glcm_correlation", "glcm_energy",
        "glcm_homogeneity", "glcm_entropy", "vario_slope", "fractal_dim",
    ]

    def __init__(self, ckpt_path: Path, scaler_path: Path, reference_path: Path):
        import csv
        import json as _json

        import torch
        import torch.nn as nn

        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        if list(ckpt["feature_cols"]) != self.FEATURE_COLS:
            raise ValueError(
                f"AE checkpoint feature_cols mismatch: {ckpt['feature_cols']}"
            )
        act_cls = {"ReLU": nn.ReLU, "GELU": nn.GELU, "SiLU": nn.SiLU}[
            ckpt["activation"]
        ]
        SpineFeatureAE = _build_spine_feature_ae_class()
        self._model = SpineFeatureAE(
            ckpt["input_dim"], ckpt["hidden"], ckpt["latent_dim"], ckpt["dropout"],
            activation=act_cls,
        )
        self._model.load_state_dict(ckpt["state_dict"])
        self._model.eval()
        self._torch = torch

        scaler = _json.loads(scaler_path.read_text(encoding="utf-8"))
        if list(scaler["features"]) != self.FEATURE_COLS:
            raise ValueError(f"AE scaler feature mismatch: {scaler['features']}")
        self._mean = np.asarray(scaler["mean"], dtype=np.float32)
        self._scale = np.asarray(scaler["scale"], dtype=np.float32)

        ref_rows = []
        with reference_path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    ref_rows.append([float(row[c]) for c in self.FEATURE_COLS])
                except (KeyError, ValueError):
                    continue
        X_ref = np.asarray(ref_rows, dtype=np.float32)
        X_ref_std = (X_ref - self._mean) / self._scale
        self._reference_sorted = np.sort(self._reconstruct_err(X_ref_std))
        logger.info(
            "이상점수 AE 로드 완료: 기준 분포 n=%d, latent=%d",
            len(self._reference_sorted), ckpt["latent_dim"],
        )

        # SHAP(KernelExplainer)용 배경 표본 (노트북 v18 E3와 동일: 원시(비표준화)
        # 특징 50개를 고정 시드로 샘플). Explainer 자체는 첫 explain() 호출까지
        # 지연 생성 — shap import를 실제 쓰는 배포에서만 부담하게 한다.
        rng = np.random.default_rng(42)
        bg_n = min(50, len(X_ref))
        self._shap_background = X_ref[
            rng.choice(len(X_ref), size=bg_n, replace=False)
        ]
        self._shap_explainer = None

    def _reconstruct_err(self, X_std: np.ndarray) -> np.ndarray:
        with self._torch.no_grad():
            x = self._torch.tensor(X_std, dtype=self._torch.float32)
            recon, _ = self._model(x)
            err = ((recon - x) ** 2).mean(dim=1).numpy()
        return err

    def score(self, feature_row: dict) -> tuple[float | None, float | None]:
        """feature_row의 FEATURE_COLS가 모두 있어야 채점된다 — 텍스처/BAR
        일부가 QC 등으로 비어 있으면 조용히 (None, None)을 반환한다."""
        vals = [feature_row.get(k) for k in self.FEATURE_COLS]
        if any(v is None for v in vals):
            return None, None
        x = np.asarray(vals, dtype=np.float32)
        x_std = (x - self._mean) / self._scale
        err = float(self._reconstruct_err(x_std[None, :])[0])
        pct = (
            100.0
            * float(np.searchsorted(self._reference_sorted, err))
            / len(self._reference_sorted)
        )
        return round(err, 5), round(pct, 1)

    def explain(self, feature_row: dict, nsamples: int = 100) -> dict[str, float] | None:
        """SHAP(KernelExplainer)로 이 척추의 이상점수(재구성오차)에 8개 특징이
        각각 얼마나 기여했는지 분해한다 (노트북 v18 E3, v18_spec.md Sec.5).
        feature_row가 불완전하면(score()가 채점 불가한 것과 동일 조건) None.
        양수 SHAP값 = 그 특징이 오차를 키운 방향(더 '비정상'쪽 기여),
        음수 = 오차를 줄인 방향(더 '정상'쪽 기여)."""
        vals = [feature_row.get(k) for k in self.FEATURE_COLS]
        if any(v is None for v in vals):
            return None
        if self._shap_explainer is None:
            import shap

            def _score_fn(X_raw: np.ndarray) -> np.ndarray:
                X_std = (X_raw - self._mean) / self._scale
                return self._reconstruct_err(X_std.astype(np.float32))

            self._shap_explainer = shap.KernelExplainer(
                _score_fn, self._shap_background
            )
        x = np.asarray([vals], dtype=np.float32)
        sv = np.asarray(
            self._shap_explainer.shap_values(x, nsamples=nsamples, silent=True)
        ).ravel()
        return {k: round(float(v), 5) for k, v in zip(self.FEATURE_COLS, sv)}


# BAR 0.50 + 텍스처 4종 가중합 (노트북 v17/v18 D0, verbatim). BAR이 절반 이상의
# 가중치를 갖는다 — 텍스처는 보조 지표, 주 지표는 여전히 밀도(BAR)라는 v18
# 설계 원칙을 그대로 반영한다.
BHI_WEIGHTS = {
    "bar": 0.50, "vario_slope": 0.20, "fractal_dim": 0.15,
    "glcm_homogeneity": 0.075, "glcm_entropy": 0.075,
}
# WHO T-score 관례를 빌린 컷포인트지만 DXA T-score가 아니라 '코호트 내 BAR
# z-score'에 적용한다 — 절대 진단이 아니라 익숙한 스케일을 빌린 것뿐이다.
BHI_CAT_LOW, BHI_CAT_VERYLOW = -1.0, -2.5
BHI_CAT_LABELS = {
    "normal": "Within reference range",
    "low": "Low (osteopenia-like)",
    "very_low": "Very low (osteoporosis-like)",
}


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(float(z) / math.sqrt(2.0)))


class _CohortReference:
    """L1~L5 코호트 기준 z-score/BHI (노트북 v17 §3 → v18 D0, verbatim 이식).

    v18_spec.md Sec.3의 명시적 요구: "각 특징의 healthy 방향(sign)은 코호트
    에서 학습, 가정 금지". BAR는 물리 법칙(감쇠가 클수록 뼈가 많다)으로
    sign=+1을 고정하고, 나머지 텍스처 특징의 부호는 이 코호트 안에서 BAR와의
    상관계수 부호로 스스로 학습한다 — 문헌 가정이 아니라 이 배포의 실측이다.

    DXA T-score가 아니라 '이 코호트 내 상대 위치'일 뿐이며, WHO 컷포인트
    (-1.0/-2.5)는 익숙한 스케일을 빌린 것이지 절대 진단이 아니다
    (v18_spec.md Sec.4 정직성 원칙 — "T-score 아님, 코호트 z-score").
    """

    def __init__(self, reference_path: Path):
        import csv

        rows: list[dict] = []
        with reference_path.open(encoding="utf-8") as f:
            rows.extend(iter(csv.DictReader(f)))
        self.reference = self._fit(rows)
        if self.reference:
            logger.info(
                "코호트 BHI 기준 적합 완료: n=%d, 특징=%s",
                self.reference["n"], sorted(self.reference["features"]),
            )

    @staticmethod
    def _fit(records: list[dict]) -> dict | None:
        ok: list[dict] = []
        for r in records:
            if r.get("qc_status") == "FAIL":
                continue
            try:
                bar = float(r["bar"])
            except (KeyError, ValueError, TypeError):
                continue
            rec = {"bar": bar}
            for f in BHI_WEIGHTS:
                if f == "bar":
                    continue
                try:
                    rec[f] = float(r[f])
                except (KeyError, ValueError, TypeError):
                    rec[f] = None
            ok.append(rec)
        if len(ok) < 3:
            return None

        features: dict[str, dict] = {}
        for f in BHI_WEIGHTS:
            pair = [
                (r[f], r["bar"]) for r in ok
                if r.get(f) is not None and np.isfinite(r[f])
            ]
            if len(pair) < 3:
                continue
            v = np.array([a for a, _ in pair], float)
            b = np.array([c for _, c in pair], float)
            sd = float(v.std(ddof=1))
            if not np.isfinite(sd) or sd < 1e-9:
                continue
            if f == "bar":
                sign = 1.0
            else:
                r_bar = float(np.corrcoef(v, b)[0, 1]) if b.std() > 0 else 0.0
                r_bar = 0.0 if not np.isfinite(r_bar) else r_bar
                sign = 1.0 if r_bar >= 0 else -1.0
            features[f] = {"mean": float(v.mean()), "sd": sd, "sign": sign}
        return {"n": len(ok), "features": features}

    def score(self, feature_row: dict) -> dict:
        """z-score/BHI/카테고리. `feature_row`는 BAR/텍스처 dict — 이상점수
        입력과 동일 딕셔너리를 그대로 재사용한다. 기준이 없거나 BAR가 없으면
        빈 dict."""
        ref = self.reference
        if not ref or not ref.get("features"):
            return {}
        zs: dict[str, float] = {}
        for f, st in ref["features"].items():
            v = feature_row.get(f)
            if v is None or not np.isfinite(v):
                continue
            zs[f] = float(st["sign"] * (v - st["mean"]) / (st["sd"] + 1e-9))
        if not zs:
            return {}

        out: dict = {}
        wsum = sum(BHI_WEIGHTS.get(f, 0.0) for f in zs)
        if wsum > 0:
            comp = sum(zs[f] * BHI_WEIGHTS.get(f, 0.0) for f in zs) / wsum
            out["bhi_z"] = round(comp, 3)
            out["bhi_pct"] = round(100.0 * _norm_cdf(comp), 1)
        if "bar" in zs:
            z = zs["bar"]
            out["bar_z"] = round(z, 3)
            out["category"] = (
                BHI_CAT_LABELS["very_low"] if z <= BHI_CAT_VERYLOW else
                BHI_CAT_LABELS["low"] if z < BHI_CAT_LOW else
                BHI_CAT_LABELS["normal"]
            )
        return out


class YoloBmdEngine(BmdInferenceEngine):
    """YOLOv8-seg 기반 L1~L5 분할 + L4 proxy BMD 산출 엔진."""

    # v17: 주 지표가 BAR(배경 대비 상대 감쇠)로 바뀌어 이전 0..1 점수와 스케일이
    # 호환되지 않는다. 버전을 올려 과거 레코드와 구분하고, 기존 스터디는 재추론한다.
    MODEL_VERSION = "yolo26m-seg-l1l5-ap-la-3.0-bar"

    def __init__(self, weights_path: Path | None = None) -> None:
        self._weights = weights_path or (
            settings.ml_model_dir / settings.ml_model_file
        )
        self._model = None  # lazy load
        self._cam = None    # lazy Seg-Grad-CAM 전용(오토그래드) 모델 인스턴스
        self._anomaly = None  # lazy AE 이상점수 스코어러
        self._bhi = None      # lazy 코호트 z-score/BHI 기준

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
        """Seg-Grad-CAM 전용 오토그래드 모델 인스턴스를 1회 로드 (실패 시 None).

        ONNX(onnxruntime) 백엔드에는 torch nn.Module/autograd가 없어 backward를
        수행할 수 없다. 이 경우 Grad-CAM(모델 주목도 크롭)은 생략하고 밀도 근거
        히트맵(xai_density.png)만 산출한다.
        """
        if self._weights.suffix.lower() == ".onnx":
            self._cam = False  # ONNX: autograd 불가 → CAM 생략
            return None
        if self._cam is None:
            try:
                self._cam = _SegGradCAM(self._weights)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Seg-Grad-CAM 초기화 실패 (XAI 히트맵 생략): %s", exc)
                self._cam = False  # 재시도 방지용 sentinel
        return self._cam or None

    def _ensure_anomaly_scorer(self):
        """AE 이상점수 스코어러 1회 로드 (실패 시 None, 이후 재시도 안 함).

        체크포인트/스케일러/기준분포 파일이 없으면(구버전 배포, ml_models
        미배치 등) 이상점수 없이 BAR/QC/텍스처만으로 정상 동작해야 한다 —
        Seg-Grad-CAM(ONNX 백엔드 미지원)과 동일한 우아한 폴백 원칙.
        """
        if self._anomaly is None:
            ckpt = settings.ml_model_dir / AE_CHECKPOINT_FILE
            scaler = settings.ml_model_dir / AE_SCALER_FILE
            reference = settings.ml_model_dir / AE_REFERENCE_FILE
            if not (ckpt.exists() and scaler.exists() and reference.exists()):
                logger.warning(
                    "AE 이상점수 아티팩트 없음 (이상점수 생략): %s", ckpt
                )
                self._anomaly = False
            else:
                try:
                    self._anomaly = _AnomalyScorer(ckpt, scaler, reference)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("AE 이상점수 스코어러 초기화 실패: %s", exc)
                    self._anomaly = False
        return self._anomaly or None

    def _ensure_bhi_reference(self):
        """코호트 z-score/BHI 기준 1회 적합 (실패 시 None, 이후 재시도 안 함).

        AE 이상점수와 같은 기준 CSV(ae_reference_features.csv, PASS-QC
        코호트)로 적합한다 — 노트북 D0가 명시한 대로 "BHI의 z-score와 AE의
        이상점수가 같은 '정상'의 정의를 공유"해야 하기 때문이다. 파일이
        없으면(구버전 배포 등) BHI 없이 BAR/QC/텍스처만으로 정상 동작한다.
        """
        if self._bhi is None:
            reference = settings.ml_model_dir / AE_REFERENCE_FILE
            if not reference.exists():
                logger.warning("BHI 기준 코호트 파일 없음 (BHI 생략): %s", reference)
                self._bhi = False
            else:
                try:
                    cohort_ref = _CohortReference(reference)
                    self._bhi = cohort_ref if cohort_ref.reference else False
                except Exception as exc:  # noqa: BLE001
                    logger.warning("BHI 기준 적합 실패: %s", exc)
                    self._bhi = False
        return self._bhi or None

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
    def _draw_bar_panel(cls, ax, rgb, panel: dict | None, label: str):
        """축(ax) 하나에 척추 하나의 BAR 히트맵 패널을 그린다 (노트북 v18 E2
        스타일). L1~L5 다중 패널(`_build_bar_maps_l1l5`)과 단일 L4 패널
        (`_build_density_heatmap`)이 이 렌더링을 공유한다.

        `panel`은 None(미검출), {"bar_map": None, ...}(BAR 산출 실패), 또는
        {"bar": float|None, "bar_map": ndarray, "roi": ndarray, "box": tuple}
        세 경우를 구분해 그린다. Returns the imshow AxesImage (컬러바 부착용)
        또는 None(컬러바가 필요 없는 경우)."""
        ax.axis("off")
        if panel is None:
            ax.set_title(f"{label}: not detected", fontsize=10)
            return None
        cx1, cy1, cx2, cy2 = panel["box"]
        bar_val = panel["bar"]
        if panel["bar_map"] is None:
            ax.imshow(rgb[cy1:cy2, cx1:cx2])
            ax.set_title(f"{label}: BAR map unavailable", fontsize=10)
            return None
        crop_gray = rgb[cy1:cy2, cx1:cx2]
        crop = panel["bar_map"][cy1:cy2, cx1:cx2]
        roi_crop = panel["roi"][cy1:cy2, cx1:cx2]
        norm, cmap = cls._bar_style(crop[roi_crop > 0])
        # PET/CT식 반투명 퓨전: 뼈 크롭을 먼저 그리고, BAR 컬러는 ROI 안에서만
        # 반투명하게 얹는다(그 바깥은 순수 뼈 영상 그대로) — 값만 보이던 단색
        # 블록 대신 해면골 질감이 색 아래로 비쳐 보이게 한다. 색 필드는 가벼운
        # 블러로 픽셀 단위 노이즈를 죽여 매끈한 등고선처럼 보이게 한다(값 자체는
        # bar_val 계산에 쓰이지 않고 오직 이 시각화에만 영향).
        crop_smooth = cv2.GaussianBlur(crop, (0, 0), sigmaX=1.2)
        # 원본 DICOM 해상도가 낮은 스터디는 이 크롭이 figure보다 작아 확대돼
        # 그려진다. imshow 기본 interpolation('antialiased')은 확대(upsampling)
        # 시 'nearest'와 동일하게 동작해 픽셀이 큼직한 사각형 덩어리로 보인다
        # ("픽셀이 너무 크다" 리포트) — bilinear로 명시해 부드럽게 보간한다.
        ax.imshow(crop_gray, interpolation="bilinear")
        alpha_mask = np.where(roi_crop > 0, 0.55, 0.0).astype(np.float32)
        im = ax.imshow(crop_smooth, cmap=cmap, norm=norm, alpha=alpha_mask,
                        interpolation="bilinear")
        ax.contour(roi_crop, levels=[0.5], colors="white", linewidths=1)
        ax.set_title(
            f"{label}: BAR={bar_val:.3f}" if bar_val is not None else f"{label}: BAR=n/a",
            fontsize=10,
        )
        return im

    @classmethod
    def _build_density_heatmap(cls, rgb, bar_map, roi, box, bar_val, out_path, label="L4"):
        """충실한 BMD 근거: L4 ROI를 BAR(배경 대비 상대 감쇠)로 색칠한 단일 패널.

        예전엔 (dens-low)/(high-low)를 JET로 칠했으나, 그 원시 감쇠 스케일은
        스터디마다 달라 색이 무엇을 의미하는지 자체로는 알 수 없었다. 지금은
        `_draw_bar_panel`(노트북 v18 E2와 동일: 0=연부조직 중심 diverging/
        sequential 컬러맵 + 컬러바 + 흰 ROI 윤곽선)을 L1~L5 다중 패널
        (`_build_bar_maps_l1l5`)과 그대로 공유해, L4 하나만 볼 때도 같은 형태로
        보이게 한다. PNG를 직접 `out_path`에 저장한다(반환값 없음).
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # "Extracted L4 vertebra" 크롭(l4_crop.png)과 화면에서 같은 크기로
        # 보이려면 figure 종횡비가 box의 실제 종횡비를 따라가야 한다 —
        # 고정 (4.6,4.8)은 box가 정사각형에 가깝다고 가정해, 더 넓적한
        # box에서는 뼈 그림이 작게 들어가고 여백만 커 보였다(리포트로 발견).
        # 컬러바는 오른쪽에 두면 ax 폭을 갉아먹어, 같은 폭으로 맞춘
        # l4_crop.png보다 뼈 그림이 좁게 나온다(리포트: "두 이미지 크기가 안
        # 맞는다") — 아래쪽 가로 컬러바로 바꿔 ax 폭은 그대로 두고 높이만
        # 늘려서 크롭과 이미지 폭이 맞도록 한다.
        cx1, cy1, cx2, cy2 = box
        w, h = max(cx2 - cx1, 1), max(cy2 - cy1, 1)
        fig_h = 4.6
        fig_w = max(3.0, fig_h * (w / h))
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        panel = {"bar": bar_val, "bar_map": bar_map, "roi": roi, "box": box}
        im = cls._draw_bar_panel(ax, rgb, panel, label)
        if im is not None:
            cbar = plt.colorbar(
                im, ax=ax, orientation="horizontal",
                fraction=0.06, pad=0.06, shrink=0.9,
            )
            cbar.ax.tick_params(labelsize=8)
        fig.tight_layout()
        fig.savefig(out_path, dpi=130, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)

    @classmethod
    def _build_air_soft_heatmap(
        cls, gray_lin, masks_xy, l4_i, valid, bone_dil, l4_window, view,
        bar_val, bar_parts, out_path,
    ):
        """BAR의 두 기준값(연부조직·air)이 실제로 어느 픽셀에서 왔는지 + 계산식을
        한 장에 보여주는 XAI 패널 (scripts/debug_air_soft_combined_visualize.py와
        동일 로직, 프로덕션용으로 이식).

        다른 XAI 패널(밀도 히트맵 등)은 L4로 좁게 크롭하지만, 이건 전체 이미지를
        보여준다 — air 배경은 대개 이미지 가장자리에, 연부조직 링은 L4 바로
        옆에 있어서 둘 다 보이려면 전체 프레임이 필요하다.

        색: 노랑=air(실제 사용), 남색=air(그림자로 제외), 초록=연부조직(채택),
        하늘색=연부조직(버려진 쪽), 빨강 채움=L4(target), 파랑 윤곽선=나머지 척추.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        H, W = gray_lin.shape[:2]

        # ---- air: _air_reference()와 동일 로직 (테두리-연결성 + 그림자/직접노출) ----
        u8 = (np.clip(gray_lin, 0, 1) * 255).astype(np.uint8)
        thr, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bg_mask_raw = u8 < thr
        bg_mask = cls._border_connected_mask(bg_mask_raw)
        bg = gray_lin[bg_mask]
        air_direct = np.zeros((H, W), dtype=bool)
        air_shadow = np.zeros((H, W), dtype=bool)
        if bg.size >= 0.005 * gray_lin.size:
            bg_only = np.where(bg_mask, u8, 0).astype(np.float32)
            local_mean = cv2.blur(bg_only, (5, 5))
            local_sq_mean = cv2.blur(bg_only ** 2, (5, 5))
            local_std = np.sqrt(np.clip(local_sq_mean - local_mean ** 2, 0, None))
            direct = bg_mask & (local_std > cls.AIR_SHADOW_STD_MAX)
            if direct.sum() >= 0.005 * gray_lin.size:
                air_direct = direct
                air_shadow = bg_mask & ~direct
            else:
                air_shadow = bg_mask
        elif bg_mask_raw.any():
            air_shadow = bg_mask_raw

        # ---- soft tissue: L4 창(pad=0.6) 안, 유효 조직, 뼈 아닌 곳 ----
        cx1, cy1, cx2, cy2 = l4_window
        window = np.zeros((H, W), dtype=bool)
        window[cy1:cy2, cx1:cx2] = True
        soft = valid & window & (bone_dil == 0)

        def _pct(mask):
            vals = gray_lin[mask]
            return (
                float(np.percentile(vals, cls.SOFT_TISSUE_PCT))
                if vals.size >= 50 else None
            )

        overlay = cv2.cvtColor(
            (np.clip(gray_lin, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB
        )
        overlay[air_shadow] = (
            0.55 * overlay[air_shadow] + 0.45 * np.array([40, 40, 160])
        ).astype(np.uint8)
        overlay[air_direct] = (
            0.4 * overlay[air_direct] + 0.6 * np.array([255, 230, 40])
        ).astype(np.uint8)

        if view == "AP":
            overlay[soft] = (
                0.5 * overlay[soft] + 0.5 * np.array([60, 230, 90])
            ).astype(np.uint8)
            soft_detail = "AP: pooled"
        else:
            split_x = (cx1 + cx2) // 2
            left = soft.copy(); left[:, split_x:] = False
            right = soft.copy(); right[:, :split_x] = False
            left_v, right_v = _pct(left), _pct(right)
            chosen_is_left = left_v is not None and (right_v is None or left_v <= right_v)
            chosen, rejected = (left, right) if chosen_is_left else (right, left)
            overlay[rejected] = (
                0.5 * overlay[rejected] + 0.5 * np.array([120, 200, 255])
            ).astype(np.uint8)
            overlay[chosen] = (
                0.5 * overlay[chosen] + 0.5 * np.array([60, 230, 90])
            ).astype(np.uint8)
            soft_detail = f"{view}: {'left' if chosen_is_left else 'right'} side used"

        for i, poly in enumerate(masks_xy):
            m = cls._poly_to_mask(poly, H, W, shrink=1.0)
            cnts, _ = cv2.findContours(
                m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            if i == l4_i:
                fill = overlay.copy()
                cv2.drawContours(fill, cnts, -1, (230, 60, 60), -1)
                overlay = (0.6 * overlay + 0.4 * fill).astype(np.uint8)
                cv2.drawContours(overlay, cnts, -1, (220, 30, 30), 2)
            else:
                cv2.drawContours(overlay, cnts, -1, (80, 160, 255), 1)

        if bar_val is not None and "a_soft" in bar_parts:
            a_soft = bar_parts["a_soft"]
            a_air = bar_parts["a_air"]
            margin = bar_parts["margin"]
            a_trab = bar_val * margin + a_soft  # bar_score 내부 계산의 역산
            formula_lines = [
                "BAR = (A_trab - A_soft) / (A_soft - A_air)",
                f"= ({a_trab:.4f} - {a_soft:.4f}) / ({a_soft:.4f} - {a_air:.4f}) = {bar_val:.4f}",
            ]
        else:
            formula_lines = ["BAR = n/a (soft/air margin unusable)"]

        # 세로로 긴(portrait) X-ray가 대부분이라 폭이 좁으면 긴 제목/범례/계산식
        # 줄이 캔버스 밖으로 잘려서 저장된다(뷰어가 아니라 PNG 파일 자체 문제) —
        # 최소 폭을 넉넉히 주고, 각 줄도 짧게 나눠 어떤 종횡비에서도 안전하게 한다.
        fig_w = max(7.5, W / H * 7.0)
        fig, ax = plt.subplots(figsize=(fig_w, 8.6))
        ax.imshow(overlay)
        ax.axis("off")
        ax.set_title(
            "Reference regions this BAR was computed from\n"
            "yellow = air (used)   navy = air (shadow, excluded)\n"
            "green = soft-tissue (used)   sky-blue = soft-tissue (rejected)\n"
            f"red fill = L4 (target)   [{soft_detail}]",
            fontsize=9,
        )
        n_formula = len(formula_lines)
        for i, line in enumerate(formula_lines):
            fig.text(
                0.5, 0.01 + (n_formula - 1 - i) * 0.028,
                line, ha="center", va="bottom", fontsize=9, family="monospace",
            )
        fig.tight_layout(rect=[0, 0.02 + 0.03 * n_formula, 1, 1])
        fig.savefig(out_path, dpi=120)
        plt.close(fig)

    @classmethod
    def _build_l4_cam_crop(cls, rgb, cam, box):
        """분할 모델 주목도: L4 크롭에 Seg-Grad-CAM을 입힌 어트리뷰션(노트북 v18 E1).

        cam(저해상도)을 원본 크기로 확대 후 L4 크롭 영역만 잘라 정규화 →
        JET 컬러맵 → 깨끗한 크롭과 블렌드. 크롭으로 한정하므로 히트맵이
        항상 L4 위에 있고 내부 골소주(trabecular) 패턴 주목까지 드러난다.
        """
        H, W = rgb.shape[:2]
        cam_full = cv2.resize(cam, (W, H))
        cx1, cy1, cx2, cy2 = box
        crop_rgb = rgb[cy1:cy2, cx1:cx2]
        crop_cam = cam_full[cy1:cy2, cx1:cx2]
        # 크롭 안에서 다시 min-max 정규화하면 안 된다. L4 내부 주목도가 평탄할 때
        # 미세 노이즈가 전체 스케일로 증폭돼, 인접 척추/피질 경계가 걸린 크롭
        # 모서리가 '가장 뜨거운 근거'처럼 보이는 착시가 생긴다(실제 관측된 버그).
        # cam은 이미 전체 이미지 기준 0..1이므로 그 스케일을 그대로 쓴다.
        crop_cam = np.clip(crop_cam, 0.0, 1.0)
        crop_heat = cv2.cvtColor(
            cv2.applyColorMap((crop_cam * 255).astype(np.uint8), cv2.COLORMAP_JET),
            cv2.COLOR_BGR2RGB,
        )
        blend = (
            (1 - CAM_BLEND) * crop_rgb + CAM_BLEND * crop_heat
        ).astype(np.uint8)
        return cls._upscale(blend)

    @classmethod
    def _bar_style(cls, vals: np.ndarray):
        """BAR 히트맵 컬러 스케일: 모든 스터디에 고정된 같은 기준
        (BAR_COLOR_VMIN..BAR_COLOR_VMAX, 컬러맵 'jet' 고정)을 쓴다.
        Grad-CAM 패널(xai_l4_cam.png, cv2.COLORMAP_JET)과 같은 톤으로 맞춰
        두 XAI 패널을 나란히 볼 때 색 언어가 일관되게 보이도록 했다.

        예전엔 ROI의 1~99 percentile로 스터디마다 스케일을 다시 잡고,
        음수를 걸치면 diverging('RdBu_r')으로 컬러맵까지 바뀌었다 —
        그 결과 같은 색이 스터디마다 다른 BAR 값을 의미했다(리포트: "기준이
        자꾸 바뀐다"). 지금은 값이 비어 있을 때만 fallback으로 None을
        반환하고, 그 외엔 항상 같은 Normalize/컬러맵을 반환해 스터디 간
        색 비교가 가능하게 한다."""
        from matplotlib.colors import Normalize

        v = vals[np.isfinite(vals)]
        if v.size == 0:
            return None, "jet"
        return Normalize(vmin=cls.BAR_COLOR_VMIN, vmax=cls.BAR_COLOR_VMAX), "jet"

    @classmethod
    def _build_bar_maps_l1l5(cls, rgb, panels_by_label: dict, out_path: str) -> None:
        """L1~L5 척추별 BAR 기여도 히트맵을 한 장의 그림으로 (노트북 v18 E2 이식).

        모든 패널이 `_bar_style`의 고정 컬러 스케일(BAR_COLOR_VMIN..VMAX)을
        공유하므로, 척추 간/스터디 간 색이 그대로 비교 가능하다. 흰색 윤곽선은
        BAR 숫자가 실제로 평균낸 trabecular ROI 그 자체다.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n = len(CLASS_NAMES)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.4))
        axes = np.atleast_1d(axes)
        for ax, label in zip(axes, CLASS_NAMES):
            im = cls._draw_bar_panel(ax, rgb, panels_by_label.get(label), label)
            if im is not None:
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(
            "Per-pixel BAR contribution (white contour = trabecular ROI actually "
            "averaged), L1-L5",
            fontsize=12,
        )
        fig.tight_layout()
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        plt.close(fig)

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
        """[레거시 v16 이전] L4 ROI 평균 감쇠를 [low, high] → 0..1 점수로 사상.

        v17부터 주 지표는 _bar_score(BAR)이며, 이 함수는 BAR 산출이 QC로 실패했을
        때의 폴백으로만 쓰인다. 두 산식은 스케일이 다르므로 값을 섞어 추세를
        그리면 안 된다(폴백 시 reliability_warning으로 표시).
        """
        score = (mean_dens - low) / (high - low)
        return round(float(np.clip(score, 0.0, 1.0)), 3)

    # ---------- v17/v18: BAR (노트북 L4_AP_segmentation_v18과 동일 산식) ----------
    DEFAULT_RESPONSE = "log"      # DICOM 태그가 없을 때 (processed CR/DX 관례)
    # v18 fix: 0.02는 A_soft가 A_air에 거의 닿을 만큼 붕괴된 마진도 통과시켜,
    # 정상적인 분자를 거의 0에 가까운 분모로 나눈 비현실적 BAR(v18 노트북 실측
    # 사례: BAR=16.8)를 "측정 신뢰 가능"으로 보고하는 사고가 있었다. 0.05로 상향.
    QC_MIN_SOFT_MARGIN = 0.05     # A_soft - A_air 가 이보다 작으면 기준값 사용 불가
    # QC 게이트 (노트북 v18과 동일 취지)
    # 연부조직 기준값 분위수. 링 오염은 단측(감쇠를 더하기만) 이므로 하위 분위수가
    # 강건 추정량이다. 25 = 링의 하위 1/4이 순수 연부조직이라고 보는 보수적 설정.
    SOFT_TISSUE_PCT = 25
    # LA에서 좌/우 후보 차이가 이보다 작으면 '더 낮은 쪽'을 확정 선택하지 않고
    # 평균을 쓴다 (scripts/debug_soft_tissue_lr_survey.py로 측면 173건 실측:
    # 28.3%(49건)가 이 문턱 미만 — 어느 쪽을 골라도 의미 없는 차이인데 매번
    # 이분법으로 한쪽을 확정하면, 재촬영/재분할 때 노이즈만으로 다른 쪽이
    # 뽑혀 기준값이 흔들릴 수 있다는 리포트로 발견).
    SOFT_TISSUE_LR_TIE_EPS = 0.02
    QC_MAX_ROI_SATURATED = 0.02   # ROI의 2% 초과가 클리핑 -> 측정 신뢰 불가
    QC_MIN_ROI_PX = 300           # 해면골 ROI가 이보다 작으면 불안정
    # v18 fix: margin 임계값이 못 잡는 경우의 독립적인 방어선. margin이 왜
    # 붕괴됐든(임계값을 살짝 통과한 경우 포함) BAR 절대값 자체가 비현실적이면
    # 뼈 판독이 아니라 파이프라인 인공물로 본다. 플레이스홀더 값 — 실제 DXA
    # 라벨이 쌓이면 재보정 필요(노트북 v18 QC_BAR_WARN_ABS/FAIL_ABS와 동일 취지).
    QC_BAR_WARN_ABS = 3.0
    QC_BAR_FAIL_ABS = 6.0
    # 히트맵 컬러 스케일 고정 기준(모든 스터디 공통). 예전엔 스터디마다 ROI의
    # 1~99 percentile로 다시 스케일을 잡아서 같은 색이 스터디마다 다른 BAR 값을
    # 뜻했다(사용자 리포트: "기준이 자꾸 바뀐다"). 0..1로 잡아 Grad-CAM
    # 패널(0=Low, 1=High attention)과 스케일까지 맞춘다 — 0(연부조직)이
    # jet의 순수 파랑, 1이 순수 빨강에 오도록. 플레이스홀더 범위 — 실제
    # 코호트 분포가 쌓이면 재보정 필요(QC_BAR_WARN_ABS와 같은 취지).
    #
    # 2026-08-13: Tried recalibrating VMAX twice (0.87 = p95, then 0.58 = p75,
    # both measured on scripts/debug_bar_distribution.py over
    # dataset-dcm/{ap,lateral}, n=213) but reverted to 1.0 at the user's
    # request. No rescaling of the color scale changes whether an
    # individual file's *measured* BAR is actually correct — a genuinely
    # low file (e.g. 3dc5b4f91af6ee03738d103a35924fba.dicom, BAR=0.04)
    # keeps looking blue and the high ones just get redder, regardless of
    # VMAX (user: "the red ones just get redder, it's meaningless"). Checked
    # that file with debug_l4_segmentation.py — the ROI is correctly placed
    # on the L4 vertebral body, and a_trab-a_soft is genuinely tiny (0.012):
    # this is a real low bone/soft-tissue-contrast image, not a code bug.
    # Next step under discussion is a low-contrast QC flag, not further
    # color-scale tuning.
    BAR_COLOR_VMIN = 0.0
    BAR_COLOR_VMAX = 1.0
    # _air_reference: Otsu 배경 안에서 '조준(collimation) 차폐 그림자'(광자
    # 미도달 -> 평탄, 국소 표준편차 거의 0)와 '진짜 직접노출'(광자 통계 잡음
    # 있음)을 가르는 국소(5x5) 표준편차 임계값(8bit 스케일). scripts/
    # debug_air_reference.py 실측(측면 2건: 배경의 67~82%가 그림자, a_air가
    # 0.09~0.14 낮게 잡힘)으로 확인 후 반영. 임계값 자체는 여전히 placeholder
    # — 표본이 쌓이면 재보정 필요(QC_BAR_WARN_ABS와 같은 취지).
    AIR_SHADOW_STD_MAX = 2.0

    # 노트북 v17 Step 3a 상수 (trabecular_roi)
    # 0.10/0.18은 몇몇 실측 사례(특히 측면)에서 여전히 침식 후에도 ROI 테두리가
    # 척추체 경계 밖(디스크 공간/인접 구조)까지 걸쳐 있었다("해면골 추출이
    # 정교하지 않다" 리포트). ROI는 결국 YOLO 분할 폴리곤 경계 안쪽만 깎아낼
    # 수 있으므로, 폴리곤 자체가 느슨하면 침식을 늘려도 한계가 있다 — 다만
    # 0.15/0.22까지는 QC_MIN_ROI_PX(300px) 대비 여유가 충분해(대표 L4 박스
    # 크기로 시뮬레이션: 가장 작은 케이스도 침식 후 3000px 이상 남음) 안전하게
    # 더 깎을 수 있다.
    ERODE_FRAC = 0.15      # 침식 커널 = 척추체 가로/세로 각 치수의 15% (피질골 테두리 제거)
    ENDPLATE_FRAC = 0.22   # 상/하 종판 밴드를 각각 22%씩 제거

    @classmethod
    def _trabecular_roi(cls, mask: np.ndarray) -> np.ndarray:
        """중앙 해면골 ROI — 노트북 v17 trabecular_roi와 동일.

        (1) 치밀한 피질골 테두리를 침식으로 제거하고, (2) 가장 밝고 오염이 심한
        상/하 종판 밴드를 잘라낸다. 남는 것은 순수 해면골(피질·종판·투영 경계 없음).
        침식이 과해 남는 게 없으면 원본 마스크를 그대로 돌려준다.

        침식 커널은 가로/세로를 각자의 실제 치수(너비/높이)에 비례해 따로
        잡는다 — 예전엔 커널을 정사각형(k x k)으로 높이에서만 구해 좌우에도
        그대로 썼는데, 척추체는 보통 세로보다 가로가 넓어 좌우 피질골이
        충분히 깎이지 않고 남는 사례가 있었다("해면골을 더 정교하게" 리포트로
        발견). 가로/세로를 각자의 치수 기준으로 침식해야 좌우 피질 여백도
        상하와 같은 비율로 제거된다.

        입력은 shrink 적용 전 '전체 L4 마스크'여야 한다(이중 축소 방지).
        """
        m = (mask > 0).astype(np.uint8)
        ys, xs = np.where(m)
        if len(xs) == 0:
            return mask
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
        h = y1 - y0 + 1
        w = x1 - x0 + 1
        ky = max(3, int(round(cls.ERODE_FRAC * h)))
        kx = max(3, int(round(cls.ERODE_FRAC * w)))
        er = cv2.erode(m, np.ones((ky, kx), np.uint8))
        cut = int(round(cls.ENDPLATE_FRAC * h))
        er[: y0 + cut] = 0
        er[y1 - cut + 1 :] = 0
        if er.sum() < 10:          # 너무 공격적 -> 안전 폴백
            return mask
        return er

    @staticmethod
    def _gray_lin(dens: np.ndarray) -> np.ndarray:
        """노트북 load_xray와 동일한 0.5/99.5 percentile 윈도우 [0,1] (뼈 밝음).

        dens는 이미 MONOCHROME1 보정이 끝난 선형 배열이므로, 노트북의 gray_lin과
        정확히 같은 값이 나온다(동일 윈도우, 동일 방향).
        """
        lo, hi = np.percentile(dens, 0.5), np.percentile(dens, 99.5)
        if hi <= lo:
            lo, hi = float(dens.min()), float(dens.max()) + 1e-6
        return np.clip((dens - lo) / (hi - lo), 0, 1).astype(np.float32)

    @classmethod
    def _detector_response(cls, path: Path) -> str:
        """DICOM PixelIntensityRelationship(LIN/LOG) → 감쇠 매핑 선택 (헤더만 읽음)."""
        try:
            import pydicom

            ds = pydicom.dcmread(str(path), stop_before_pixels=True)
            pir = str(getattr(ds, "PixelIntensityRelationship", "")).strip().upper()
            if pir == "LIN":
                return "linear"
            if pir == "LOG":
                return "log"
        except Exception:  # noqa: BLE001 - 헤더 불량 시 기본값으로 진행
            pass
        return cls.DEFAULT_RESPONSE

    @classmethod
    def _view_position(cls, path: Path) -> str:
        """DICOM view-position tag -> "AP"|"LA"|"unknown" (header only, no
        pixel data read).

        Used by soft_tissue_ref() to pick its soft-tissue band strategy
        (notebook v18, v18_spec.md Sec.4.5). Checks the standard
        ViewPosition tag first, then falls back to free-text
        SeriesDescription. Returns "unknown" if both are ambiguous — this
        used to default to "AP" instead (on the assumption that most
        untagged studies are AP), but measuring the real dataset
        (scripts/debug_view_features.py, dataset-dcm/{ap,lateral}, AP n=41
        + LA n=174) showed this cohort's DICOM anonymization strips
        ViewPosition on lateral studies too — defaulting to "AP" would have
        mis-registered all 174 lateral studies (user-confirmed: "every file
        in the lateral folder really is lateral"). Returning "unknown"
        instead lets the caller fall through to `_view_from_shape()` (the
        shape-based second tier v18_spec.md originally specified).
        """
        try:
            import pydicom

            ds = pydicom.dcmread(str(path), stop_before_pixels=True)
            tag = str(getattr(ds, "ViewPosition", "") or "").strip().upper()
            desc = str(getattr(ds, "SeriesDescription", "") or "").strip().upper()
        except Exception:  # noqa: BLE001 - 헤더 불량 시 안전한(unknown) 기본값
            return "unknown"

        if tag in ("AP", "PA"):
            return "AP"
        if tag in ("LL", "RL", "LAT", "LATERAL"):
            return "LA"
        if "LATERAL" in desc or "LAT" in desc:
            return "LA"
        if "AP" in desc or "PA" in desc:
            return "AP"
        return "unknown"

    @classmethod
    def _view_from_shape(
        cls, boxes: np.ndarray, masks_xy, H: int, W: int
    ) -> str:
        """AP/LA classification from the shape of the YOLO vertebra
        segmentation — the second-tier fallback for when the DICOM tag is
        missing/ambiguous (`_view_position()` -> "unknown"). This is the
        "aspect-ratio / shape heuristic" step v18_spec.md Sec.4.5 always
        specified but that was never actually implemented (the old code
        jumped straight from "no tag" to a hardcoded "AP" default).

        Measured on scripts/debug_view_features.py over the real dataset
        (dataset-dcm/{ap,lateral}, AP n=41 + LA n=174 — both cohorts with
        DICOM-tag/user-confirmed ground truth):
          - vert_ar_mean (mean per-vertebra bounding-box width/height) was
            the strongest single feature: AP median=2.14 (p25=1.93) vs
            LA median=1.25 (p75=1.33) — the IQRs barely overlap. In AP the
            box spans the vertebral body's full transverse width (wide), in
            LA it only spans the body's front-to-back depth (closer to
            square).
          - Adding symmetry_iou (IoU of the combined vertebra mask against
            its own left-right mirror, cropped to its bounding box) as a
            secondary feature, the score `vert_ar_mean + 0.5*symmetry_iou`
            with threshold 1.88 got 214/215 (99.5%) correct — far better
            than the old always-"AP" default's measured accuracy on this
            same set (41/215 = 19.1%).
        `boxes` is the xyxy pixel-coordinate array, `masks_xy` the list of
        per-vertebra polygon coordinates (YOLO's `r.boxes.xyxy`/`r.masks.xy`)
        — reused as-is from the caller's already-completed segmentation, no
        extra inference. If this misclassifies, the physician can manually
        set AP/LA on screen and recompute (view_source="manual").
        """
        n = len(boxes)
        if n == 0:
            return "AP"

        bone_union = np.zeros((H, W), np.uint8)
        vert_ars: list[float] = []
        for i in range(n):
            bone_union |= cls._poly_to_mask(masks_xy[i], H, W, shrink=1.0)
            bw = float(boxes[i][2] - boxes[i][0])
            bh = float(boxes[i][3] - boxes[i][1])
            if bh > 0:
                vert_ars.append(bw / bh)
        if not vert_ars:
            return "AP"
        vert_ar_mean = float(np.mean(vert_ars))

        ys, xs = np.where(bone_union > 0)
        if xs.size == 0:
            return "AP"
        bx0, bx1 = int(xs.min()), int(xs.max())
        by0, by1 = int(ys.min()), int(ys.max())
        crop = bone_union[by0 : by1 + 1, bx0 : bx1 + 1] > 0
        flipped = crop[:, ::-1]
        union_px = np.logical_or(crop, flipped).sum()
        symmetry_iou = (
            float(np.logical_and(crop, flipped).sum()) / float(union_px)
            if union_px > 0
            else 0.0
        )

        score = vert_ar_mean + 0.5 * symmetry_iou
        return "AP" if score > 1.88 else "LA"

    @staticmethod
    def _to_attenuation(gray: np.ndarray, response: str) -> np.ndarray:
        """[0,1] '뼈 밝음' 영상 → 감쇠 비례 도메인.

        log    : 저장값이 이미 mu*t에 아핀 → A = I
        linear : 저장값이 투과 강도에 비례 → Beer-Lambert로 A = -ln(1 - I)
        둘 다 감쇠에 대해 단조증가이므로 '밝다 = 밀도 높다'가 유지된다.
        """
        if response == "log":
            return np.asarray(gray, np.float32)
        g = np.clip(np.asarray(gray, np.float32), 0.0, 1.0 - 1e-4)
        return (-np.log1p(-g)).astype(np.float32)

    @staticmethod
    def _trimmed_mean(v: np.ndarray, lo: int = 10, hi: int = 90) -> float | None:
        """중앙 80% 평균 — 잔여 종판/혈관 그림자/핫픽셀에 둔감."""
        if v.size == 0:
            return None
        a, b = np.percentile(v, [lo, hi])
        core = v[(v >= a) & (v <= b)]
        return float(core.mean()) if core.size else float(v.mean())

    @staticmethod
    def _border_connected_mask(mask: np.ndarray) -> np.ndarray:
        """mask 중 이미지 테두리에 닿아있는 연결성분만 남긴다.

        진짜 콜리메이션 배경은 항상 프레임 가장자리에서 시작해 몸 쪽으로
        번져오는 형태다. 반면 장내가스·폐야(허파)처럼 뼈보다 어두운 몸통
        내부 영역은 Otsu 임계값 하나만으로는 배경과 구분되지 않는다 — 밝기만
        보면 똑같이 "배경 후보"로 잡힌다. 이 둘의 결정적 차이는 위치다:
        전자는 테두리에 붙어 있고 후자는 몸통 안에 고립된 섬(island)이다.
        connectedComponents로 나눈 뒤 이미지 네 변 중 하나라도 닿아있는
        성분만 남기면, 몸통 내부에 갇힌 성분은 밝기와 무관하게 제외된다
        (scripts/debug_air_visualize.py로 176장 실측: air 값이 비정상적으로
        높게 (0.3~0.6대) 나온 파일들은 하나같이 복부 장내가스나 흉곽 늑골
        사이 영역이 배경으로 오분류된 경우였다 — "종판/disc 대신 air 자체를
        시각화해보니 몸통 안쪽이 초록(직접노출 판정)으로 찍힌다"는 리포트로
        발견).
        """
        m8 = mask.astype(np.uint8)
        n, lbl = cv2.connectedComponents(m8, connectivity=8)
        if n <= 1:
            return np.zeros_like(mask)
        border_labels = set(np.unique(lbl[0, :])) | set(np.unique(lbl[-1, :]))
        border_labels |= set(np.unique(lbl[:, 0])) | set(np.unique(lbl[:, -1]))
        border_labels.discard(0)
        if not border_labels:
            return np.zeros_like(mask)
        return np.isin(lbl, list(border_labels))

    @classmethod
    def _air_reference(cls, gray: np.ndarray) -> float:
        """직접노출(비감쇠) 배경 레벨. Otsu로 배경을 분리하고, 테두리에 닿은
        연결성분만 남긴 뒤(`_border_connected_mask`, 몸통 내부의 장내가스/
        폐야 오염 배제), 그 안에서 다시 '조준(collimation) 차폐 그림자'와
        '진짜 직접노출'을 갈라 후자의 25 percentile을 쓴다.

        차폐 그림자는 광자가 아예 도달하지 않은 영역이라 평탄하고(국소 표준편차
        ~0), 직접노출은 광자가 도달했으므로 낮은 값이라도 포아송 잡음이
        남는다. 이 둘을 구분 없이 배경으로 묶으면(예전 방식) 그림자 쪽이
        진짜 직접노출보다 항상 더 낮으므로 a_air가 실제보다 낮게 잡히고,
        margin(a_soft-a_air)이 부풀어 BAR 전체가 0 쪽으로 눌린다. 측면(LA)
        촬영은 산란선/피폭 저감을 위해 조준을 좁게 잡는 경우가 많아 배경
        대부분이 그림자인 사례가 흔하다(scripts/debug_air_reference.py로 측면
        2건 실측: 배경의 67~82%가 그림자, a_air가 0.09~0.14 낮게 잡혀 있었음
        — "측면 BAR가 이유 없이 낮다"는 리포트로 발견).

        테두리-연결성 필터링 후 후보가 전체의 0.5% 미만이면(협착 조준 등,
        사실상 진짜 배경이 없는 경우) 필터링 전 Otsu 배경 전체의 25
        percentile로, 그마저도 없으면 전체 이미지 1 percentile로 순서대로
        폴백한다.
        """
        u8 = (np.clip(gray, 0, 1) * 255).astype(np.uint8)
        thr, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bg_mask_raw = u8 < thr
        bg_mask = cls._border_connected_mask(bg_mask_raw)
        bg = gray[bg_mask]
        if bg.size < 0.005 * gray.size:
            bg_fallback = gray[bg_mask_raw]
            if bg_fallback.size >= 0.005 * gray.size:
                return float(np.percentile(bg_fallback, 25))
            return float(np.percentile(gray, 1))

        bg_only = np.where(bg_mask, u8, 0).astype(np.float32)
        local_mean = cv2.blur(bg_only, (5, 5))
        local_sq_mean = cv2.blur(bg_only ** 2, (5, 5))
        local_std = np.sqrt(np.clip(local_sq_mean - local_mean ** 2, 0, None))
        direct_exposure = bg_mask & (local_std > cls.AIR_SHADOW_STD_MAX)
        if direct_exposure.sum() >= 0.005 * gray.size:
            return float(np.percentile(gray[direct_exposure], 25))
        return float(np.percentile(bg, 25))

    @classmethod
    def _soft_tissue_ref(cls, gray_lin, soft, valid, l4_box, view):
        """연부조직 기준값 — view-adaptive 밴드 전략 (노트북 v18 Sec.4.5).

        `soft`는 L4 주변 창에서 검출된 모든 척추체를 뺀 국소 링이다.

        - **AP**: 링 전체를 풀링해 하위 분위수(SOFT_TISSUE_PCT)를 취한다(v17.2
          그대로 — 좌우 paraspinal 근육은 해부학적으로 대칭이라 함께 풀링해도
          안전하고, 표본이 늘어 더 안정적이다).
        - **LA/unknown**: 측면 영상은 좌우가 대칭이 아니다 — 한쪽은 복측(배쪽,
          깨끗한 연부조직), 다른 쪽은 후방 골구조물(극돌기·추궁판)이 겹쳐 뼈로
          오염된다. 풀링하면 오염된 쪽이 기준값을 끌어올리므로(v17 LA 실측:
          BAR<0인 물리적으로 불가능한 사례 발생 원인), 링을 좌/우로 나눠 각각
          하위 분위수를 구한 뒤 **더 낮은(연부조직에 가까운) 쪽**을 채택한다 —
          오염은 감쇠를 더하기만 하므로, 깨끗한 쪽이 항상 더 낮게 읽힌다.

        어느 경로든 유효 표본이 50px 미만이면 유효 조직 전체의 하위 분위수로
        폴백한다(기존 동작 유지).
        """
        def _pct(vals: np.ndarray) -> float | None:
            return (
                float(np.percentile(vals, cls.SOFT_TISSUE_PCT))
                if vals.size >= 50
                else None
            )

        ref: float | None = None
        if view == "AP":
            ref = _pct(gray_lin[soft])
        else:
            cx1, _cy1, cx2, _cy2 = l4_box
            split_x = (cx1 + cx2) // 2
            left = soft.copy()
            left[:, split_x:] = False
            right = soft.copy()
            right[:, :split_x] = False
            candidates = [
                v for v in (_pct(gray_lin[left]), _pct(gray_lin[right]))
                if v is not None
            ]
            if candidates:
                ref = min(candidates)

        if ref is not None:
            return ref
        body = gray_lin[valid]
        if body.size == 0:
            return None
        return float(np.percentile(body, cls.SOFT_TISSUE_PCT))

    @classmethod
    def _bar_score(cls, gray_lin, roi_mask, valid, l4_box, bone_dil, response, view="AP"):
        """BAR = (A_trab - A_soft) / (A_soft - A_air) — 노트북 v17/v18과 동일.

        분자·분모가 모두 '차이'이므로 감쇠의 아핀 변환 A -> a*A + b (노출/게인/
        윈도우레벨 변화)에 대해 불변이다. 읽는 법: 'L4 해면골은 주변 연부조직보다
        BAR배 더 많은 추가 감쇠를 만든다'.

        반환 (bar | None, parts). parts에는 게이지/격자용 bar_map과 a_soft/a_air/
        margin이 담긴다. 연부조직 기준값이 없거나 soft-air 마진이 QC 미만이면
        bar=None → 호출부에서 레거시 점수로 폴백한다.
        """
        cx1, cy1, cx2, cy2 = l4_box
        window = np.zeros(gray_lin.shape, dtype=bool)
        window[cy1:cy2, cx1:cx2] = True
        soft = valid & window & (bone_dil == 0)

        ref = cls._soft_tissue_ref(gray_lin, soft, valid, l4_box, view)
        if ref is None:
            return None, {}

        air_gray = cls._air_reference(gray_lin)
        A = cls._to_attenuation(gray_lin, response)
        a_soft = float(cls._to_attenuation(np.array([[ref]], np.float32), response)[0, 0])
        a_air = float(cls._to_attenuation(np.array([[air_gray]], np.float32), response)[0, 0])
        margin = a_soft - a_air
        if margin <= cls.QC_MIN_SOFT_MARGIN:
            return None, {"a_soft": a_soft, "a_air": a_air, "margin": float(margin)}

        # ROI에서도 금속(인공관절)·배경을 제외한다. 레거시 경로(vals = dens[(mask>0) & valid])
        # 와 동일한 규칙. 제외 후 남는 픽셀이 없으면 원본 ROI로 폴백한다.
        roi_sel = (roi_mask > 0) & valid
        if not roi_sel.any():
            roi_sel = roi_mask > 0
        a_trab = cls._trimmed_mean(A[roi_sel])
        if a_trab is None:
            return None, {}
        bar = float((a_trab - a_soft) / margin)
        bar_map = ((A - a_soft) / margin).astype(np.float32)
        # 게이지 스케일: 0 = 연부조직, high = 유효 조직 99pct(치밀골 수준)
        gauge_high = (
            float(np.percentile(bar_map[valid], 99)) if valid.any()
            else float(bar_map[roi_mask > 0].max())
        )
        return round(bar, 3), {
            "bar_map": bar_map,
            "a_soft": a_soft,
            "a_air": a_air,
            "margin": float(margin),
            "gauge_low": 0.0,
            "gauge_high": max(gauge_high, bar + 1e-3),
        }

    # ---------- 텍스처(골소주 미세구조): GLCM/variogram/fractal (노트북 v18 B4) ----------
    @classmethod
    def _compute_glcm(cls, gray_lin, mask, levels: int = GLCM_LEVELS):
        """회전평균 GLCM 대비/상관/에너지/균질성 + 엔트로피 — 노트북과 동일 산식."""
        from skimage.feature import graycomatrix, graycoprops

        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return None
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        roi = (gray_lin[y0:y1, x0:x1] * (levels - 1)).astype(np.uint8)
        mroi = mask[y0:y1, x0:x1]
        roi[mroi == 0] = 0
        glcm = graycomatrix(
            roi, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
            levels=levels, symmetric=True, normed=True,
        )
        p = glcm.astype(np.float64)
        p = p / (p.sum(axis=(0, 1), keepdims=True) + 1e-12)
        entropy = float((-(p * np.log2(p + 1e-12)).sum(axis=(0, 1))).mean())
        return {
            "glcm_contrast": float(graycoprops(glcm, "contrast").mean()),
            "glcm_correlation": float(graycoprops(glcm, "correlation").mean()),
            "glcm_energy": float(graycoprops(glcm, "energy").mean()),
            "glcm_homogeneity": float(graycoprops(glcm, "homogeneity").mean()),
            "glcm_entropy": entropy,
        }

    @staticmethod
    def _vario_slope(gray_lin, mask, max_lag: int = VARIO_MAX_LAG) -> float | None:
        """TBS 유사 텍스처 지수: log(variogram) vs log(lag) 기울기 — 노트북과 동일."""
        ys, xs = np.where(mask > 0)
        if len(xs) < 50:
            return None
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        p = gray_lin[y0:y1, x0:x1].astype(np.float64)
        mk = mask[y0:y1, x0:x1] > 0
        lags: list[int] = []
        gam: list[float] = []
        for k in range(1, max_lag + 1):
            d = []
            m = mk[:, k:] & mk[:, :-k]
            if m.any():
                d.append(((p[:, k:] - p[:, :-k])[m]) ** 2)
            m = mk[k:, :] & mk[:-k, :]
            if m.any():
                d.append(((p[k:, :] - p[:-k, :])[m]) ** 2)
            if not d:
                continue
            v = float(np.concatenate(d).mean())
            if v > 0:
                lags.append(k)
                gam.append(v)
        if len(lags) < 3:
            return None
        s, _ = np.polyfit(np.log(lags), np.log(gam), 1)
        return float(s)

    @staticmethod
    def _fractal_dim(gray_lin, mask, block: int = FRACTAL_BLOCK) -> float | None:
        """골소주 네트워크의 box-counting 프랙탈 차원 — 노트북과 동일."""
        ys, xs = np.where(mask > 0)
        if len(xs) < 100:
            return None
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        p = gray_lin[y0:y1, x0:x1]
        mk = mask[y0:y1, x0:x1] > 0
        u8 = (np.clip(p, 0, 1) * 255).astype(np.uint8)
        b = min(block, min(u8.shape) - 1)
        b = b if b % 2 == 1 else b - 1
        if b < 3:
            return None
        bw = cv2.adaptiveThreshold(
            u8, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, b, -2
        )
        bw = (bw > 0) & mk
        n = min(bw.shape)
        if n < 16 or not bw.any():
            return None
        sizes: list[int] = []
        counts: list[int] = []
        s = 2
        while s <= n // 2:
            Hs, Ws = bw.shape[0] // s * s, bw.shape[1] // s * s
            blk = bw[:Hs, :Ws].reshape(Hs // s, s, Ws // s, s).any(axis=(1, 3))
            c = int(blk.sum())
            if c > 0:
                sizes.append(s)
                counts.append(c)
            s *= 2
        if len(sizes) < 3:
            return None
        slope, _ = np.polyfit(np.log(1.0 / np.array(sizes, float)), np.log(counts), 1)
        return float(slope)

    @classmethod
    def _compute_texture(cls, gray_lin, roi_mask) -> dict:
        """해면골 ROI 1개의 미세구조 특징 전체 (BAR와 별도 축 — AE 이상점수 입력)."""
        out: dict = {}
        g = cls._compute_glcm(gray_lin, roi_mask)
        if g:
            out.update(g)
        out["vario_slope"] = cls._vario_slope(gray_lin, roi_mask)
        out["fractal_dim"] = cls._fractal_dim(gray_lin, roi_mask)
        return out

    # ---------- v2: L1~L5 전체 척추에 대한 BAR/QC (예전엔 target(L4)만) ----------
    @classmethod
    def _vertebra_qc(
        cls, bar: float | None, roi_raw: np.ndarray, dens_max: float
    ) -> tuple[str, str | None]:
        """척추 1개의 QC 판정. 노트북 v18의 qc_check()와 동일 취지:
        _bar_score() 내부에서 이미 걸러지는 것(연부조직 기준값/soft-air 마진) 외에
        여기서는 _bar_score가 볼 수 없는 두 가지를 추가로 본다 — ROI 자체의
        포화/크기, 그리고 상류 체크를 다 통과했어도 비율 자체가 비현실적인 경우의
        절대값 방어선(QC_BAR_WARN_ABS/FAIL_ABS).

        Returns (status, message) where status is "PASS" | "WARN" | "FAIL".
        """
        hard: list[str] = []
        soft: list[str] = []
        if roi_raw.size == 0:
            hard.append("Trabecular ROI is empty.")
        else:
            if roi_raw.size < cls.QC_MIN_ROI_PX:
                hard.append(f"Trabecular ROI too small ({roi_raw.size}px).")
            sat = float((roi_raw >= 0.999 * dens_max).mean())
            if sat > cls.QC_MAX_ROI_SATURATED:
                soft.append(
                    f"{sat*100:.1f}% of ROI pixels are clipped (saturated)."
                )
        if bar is None:
            hard.append(
                "Soft-tissue reference unusable -> BAR could not be computed."
            )
        else:
            if abs(bar) >= cls.QC_BAR_FAIL_ABS:
                hard.append(
                    f"BAR={bar:.2f} is outside plausible range "
                    f"(>= {cls.QC_BAR_FAIL_ABS}) -- likely a collapsed "
                    f"soft-tissue margin, not a real bone reading."
                )
            elif abs(bar) >= cls.QC_BAR_WARN_ABS:
                soft.append(
                    f"BAR={bar:.2f} is unusually large "
                    f"(>= {cls.QC_BAR_WARN_ABS}) -- interpret with caution."
                )
        status = "FAIL" if hard else ("WARN" if soft else "PASS")
        msgs = hard + soft
        return status, "; ".join(msgs) if msgs else None

    @classmethod
    def _measure_vertebra_bar(
        cls,
        gray_lin: np.ndarray,
        dens: np.ndarray,
        poly_xy,
        H: int,
        W: int,
        valid: np.ndarray,
        bone_dil: np.ndarray,
        response: str,
        crop_box: tuple[int, int, int, int],
        view: str = "AP",
    ):
        """단일 척추의 전체 측정: trabecular ROI -> BAR -> QC.

        모든 검출 척추에 대해 재사용되는 단일 책임 함수 (v18 B의 L1~L5 일반화와
        동일 원칙 — 예전엔 이 로직이 target(L4)에만 인라인으로 존재했다).

        `bone_dil`은 검출된 '모든' 척추의 합집합(팽창)이므로, 이 척추 자신의
        본체도 이미 포함돼 있다. 따라서 별도로 '이 척추만 제외'한 마스크를
        새로 만들 필요 없이 그대로 재사용해도 연부조직 창에서 자기 자신과
        이웃 척추 모두 올바르게 제외된다.

        `view` ("AP"|"LA"|"unknown", v18 Sec.4.5)는 이미지 1장당 한 값이며
        검출된 모든 척추에 동일하게 적용된다 — soft_tissue_ref()의 밴드 전략
        선택에 쓰인다.

        Returns (bar, qc_status, qc_message, trab_roi_mask, bar_parts).
        """
        full_mask = cls._poly_to_mask(poly_xy, H, W, shrink=1.0)
        trab_roi = cls._trabecular_roi(full_mask)
        bar, bar_parts = cls._bar_score(
            gray_lin, trab_roi, valid, crop_box, bone_dil, response, view
        )
        roi_raw = dens[trab_roi > 0]
        dmax = float(dens.max())
        qc_status, qc_message = cls._vertebra_qc(bar, roi_raw, dmax)
        return bar, qc_status, qc_message, trab_roi, bar_parts

    # ---------- Precision / LSC ----------
    @staticmethod
    def _jitter(rgb, dens, deg, dx, dy, scale):
        """강체(회전+이동)+스케일 섭동. 두 방문 사이 재위치 오차를 흉내낸다
        (노트북 v18 _jitter_image와 동일 산식). rgb/dens 둘 다 같은 아핀
        변환으로 워프해야 두 배열의 픽셀 대응이 어긋나지 않는다."""
        H, W = dens.shape[:2]
        M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), deg, scale)
        M[0, 2] += dx
        M[1, 2] += dy
        rgb_j = cv2.warpAffine(
            rgb, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )
        dens_j = cv2.warpAffine(
            dens, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )
        return rgb_j, dens_j

    def _quick_target_bar(self, rgb, dens, response, view="AP") -> float | None:
        """target 척추의 BAR만 빠르게 산출 (오버레이/크롭/XAI 이미지 생성 없음).
        측정 정밀도(LSC) 산출을 위해 같은 스터디를 여러 번 재측정할 때 쓴다 —
        run()의 전체 산출물 저장 비용 없이 숫자 하나만 필요하기 때문."""
        model = self._ensure_model()
        H, W = dens.shape[:2]
        r = model.predict(rgb, conf=CONF, imgsz=IMGSZ, verbose=False)[0]
        if r.masks is None or len(r.boxes) == 0:
            return None
        boxes = r.boxes.xyxy.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()
        l4_i, _status = self._pick_l4_index(boxes, classes, confs)
        if l4_i is None:
            return None

        valid = self._valid_tissue_mask(dens)
        bone_union = np.zeros((H, W), np.uint8)
        for i in range(len(boxes)):
            bone_union |= self._poly_to_mask(r.masks.xy[i], H, W, shrink=1.0)
        bone_dil = cv2.dilate(bone_union, np.ones((7, 7), np.uint8), 1)
        gray_lin = self._gray_lin(dens)
        crop_box = self._crop_box(H, W, boxes[l4_i], pad=0.6)
        bar, qc_status, _msg, _roi, _parts = self._measure_vertebra_bar(
            gray_lin, dens, r.masks.xy[l4_i], H, W, valid, bone_dil, response, crop_box, view
        )
        if qc_status == "FAIL":
            return None
        return bar

    def repeat_measure(
        self, dicom_path: Path, n: int = LSC_N_REPEATS, seed: int = 0
    ) -> list[float]:
        """같은 스터디를 작은 재위치 섭동 하에서 n번 재측정 (노트북 v18의
        repeat_measure_v18과 동일 원칙, L4 target 한정). 첫 번째는 섭동 없는
        원본. 반환값은 QC를 통과한 BAR만 (실패/미검출은 제외)."""
        rgb0, dens0, *_ = self._load_dicom(dicom_path)
        response = self._detector_response(dicom_path)
        H, W = dens0.shape[:2]
        tag_view = self._view_position(dicom_path)
        if tag_view != "unknown":
            view = tag_view
        else:
            # If the tag is missing (_view_position -> "unknown"), run
            # segmentation once on the unperturbed original to resolve view via
            # the shape heuristic (_view_from_shape) — every repeat must use
            # the same view for the precision (LSC) measurement's "one value
            # per image" assumption to hold.
            model = self._ensure_model()
            r0 = model.predict(rgb0, conf=CONF, imgsz=IMGSZ, verbose=False)[0]
            view = (
                self._view_from_shape(r0.boxes.xyxy.cpu().numpy(), r0.masks.xy, H, W)
                if r0.masks is not None and len(r0.boxes) > 0
                else "AP"
            )
        rng = np.random.default_rng(seed)
        vals: list[float] = []
        for i in range(n):
            if i == 0:
                rgb, dens = rgb0, dens0
            else:
                rgb, dens = self._jitter(
                    rgb0, dens0,
                    deg=rng.uniform(-JITTER_DEG, JITTER_DEG),
                    dx=rng.uniform(-JITTER_SHIFT_FRAC, JITTER_SHIFT_FRAC) * W,
                    dy=rng.uniform(-JITTER_SHIFT_FRAC, JITTER_SHIFT_FRAC) * H,
                    scale=1.0 + rng.uniform(-JITTER_SCALE, JITTER_SCALE),
                )
            bar = self._quick_target_bar(rgb, dens, response, view)
            if bar is not None and np.isfinite(bar):
                vals.append(float(bar))
        return vals

    def measure_precision(
        self, dicom_paths: list[Path], n_repeats: int = LSC_N_REPEATS, seed: int = 42
    ) -> dict | None:
        """ISCD 정밀도: 여러 스터디에 걸친 RMS-CV%, LSC% = 2.77 x RMS-CV%
        (95% 신뢰수준). 노트북 v18 precision_lsc()와 동일 산식 — 차이는 합성
        데모가 아니라 이 배포에 실제 업로드된 스터디로 계산한다는 것뿐이다.
        LSC%보다 작은 변화는 재위치 노이즈이지 실제 골밀도 변화가 아니다."""
        cvs = []
        n_used = 0
        for i, path in enumerate(dicom_paths):
            try:
                vals = self.repeat_measure(path, n=n_repeats, seed=seed + i)
            except Exception as exc:  # noqa: BLE001
                logger.warning("정밀도 재측정 실패 (건너뜀) %s: %s", path, exc)
                continue
            arr = np.asarray(vals, float)
            if arr.size < 2 or abs(arr.mean()) < 1e-9:
                continue
            cvs.append(float(arr.std(ddof=1) / abs(arr.mean())))
            n_used += 1
        if not cvs:
            return None
        rms_cv = float(np.sqrt(np.mean(np.square(cvs))))
        return {
            "n_studies": n_used,
            "n_repeats": n_repeats,
            "rms_cv_pct": round(100 * rms_cv, 3),
            "lsc_pct": round(100 * LSC_Z95 * rms_cv, 3),
        }

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
    def run(
        self,
        dicom_path: Path,
        output_dir: Path,
        view_override: str | None = None,
    ) -> InferenceResult:
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

        # Seg-Grad-CAM은 별도 오토그래드 전용 모델로 자체 forward+backward를
        # 수행하므로(Eigen-CAM처럼 이 predict() 호출에 훅을 걸 필요 없음), 여기서는
        # 지연 로드만 해 둔다 — 실제 계산은 아래 XAI 산출 단계에서.
        grad_cam = self._ensure_cam()

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
        # 뼈 경계가 연부조직에 새지 않도록 뼈 영역을 살짝 팽창시켜 사용.
        bone_dil = cv2.dilate(bone_union, np.ones((7, 7), np.uint8), 1)

        # ---- v18: 전체 검출 척추(L1~L5)에 대해 BAR/QC 산출 ----
        # 예전엔 이 블록이 target(L4)에만 인라인으로 존재했다. '어떻게 재는가'는
        # 그대로 두고(노트북 L4_AP_segmentation_v18과 동일 산식) '어떤 척추까지
        # 재는가'만 L1~L5 전체로 넓힌다 (v18 B1~B4 일반화와 동일 원칙).
        # 배경(연부조직) 대비 상대 감쇠로 정규화 -> 노출/게인/윈도우레벨 불변.
        gray_lin = self._gray_lin(dens)
        response = self._detector_response(dicom_path)
        # view("AP"|"LA", v18 Sec.4.5): 이미지 1장당 한 값. soft_tissue_ref()가 AP는
        # 좌우 링을 풀링, LA는 좌우로 나눠 더 낮은(연부조직에 가까운) 쪽을 채택하도록
        # 분기한다 — 측면 영상에서 후방 골구조물이 기준값을 오염시키는 문제 대응.
        # If view_override is set (physician manually specified AP/LA on screen
        # to recompute), skip auto-detection and use it as-is. If the tag is
        # missing ("unknown"), fall through to the shape-based second tier
        # (_view_from_shape) reusing the segmentation (boxes/masks) already
        # computed above — no extra inference.
        tag_view = self._view_position(dicom_path)
        auto_view = (
            tag_view
            if tag_view != "unknown"
            else self._view_from_shape(boxes, r.masks.xy, H, W)
        )
        view = view_override or auto_view
        view_source = "manual" if view_override else "auto"
        anomaly_scorer = self._ensure_anomaly_scorer()
        bhi_reference = self._ensure_bhi_reference()
        l4_window = None
        # L1~L5 BAR 히트맵 그림(xai_bar_l1l5.png)용 — 척추 라벨별로 bar_map/ROI/box를
        # 모아둔다. target(L4)뿐 아니라 검출된 모든 척추를 담는다.
        panels_by_label: dict = {}
        for i in range(len(boxes)):
            crop_box_i = self._crop_box(H, W, boxes[i], pad=0.6)
            bar_i, qc_status_i, qc_message_i, trab_roi_i, bar_parts_i = (
                self._measure_vertebra_bar(
                    gray_lin, dens, r.masks.xy[i], H, W, valid, bone_dil,
                    response, crop_box_i, view,
                )
            )
            segments[i].bar = bar_i
            segments[i].qc_status = qc_status_i
            segments[i].qc_message = qc_message_i
            panels_by_label[segments[i].label] = {
                "bar": bar_i,
                "bar_map": bar_parts_i.get("bar_map"),
                "roi": trab_roi_i,
                "box": crop_box_i,
            }

            # ---- v3: 텍스처(골소주 미세구조) + AE 이상점수 (노트북 v18 B4/C9) ----
            # BAR와 같은 해면골 ROI(유효 조직만)에서 뽑는다 — 서로 다른 ROI를 쓰면
            # 두 지표가 '다른 것을 본' 값이 돼 이상점수의 입력 벡터가 어긋난다.
            texture_mask = ((trab_roi_i > 0) & valid).astype(np.uint8)
            texture = self._compute_texture(gray_lin, texture_mask)
            segments[i].glcm_contrast = texture.get("glcm_contrast")
            segments[i].glcm_correlation = texture.get("glcm_correlation")
            segments[i].glcm_energy = texture.get("glcm_energy")
            segments[i].glcm_homogeneity = texture.get("glcm_homogeneity")
            segments[i].glcm_entropy = texture.get("glcm_entropy")
            segments[i].vario_slope = texture.get("vario_slope")
            segments[i].fractal_dim = texture.get("fractal_dim")

            feature_row = {"bar": bar_i, **texture}
            if anomaly_scorer is not None:
                anomaly_mse, anomaly_pct = anomaly_scorer.score(feature_row)
                segments[i].anomaly_mse = anomaly_mse
                segments[i].anomaly_pct = anomaly_pct
                if anomaly_pct is not None:
                    try:
                        segments[i].anomaly_shap = anomaly_scorer.explain(
                            feature_row
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("SHAP 설명 생성 실패 (생략): %s", exc)

            # ---- v4: 코호트 z-score/BHI (노트북 v17 §3 -> v18 D0) ----
            if bhi_reference is not None:
                bhi = bhi_reference.score(feature_row)
                segments[i].bar_z = bhi.get("bar_z")
                segments[i].bhi_z = bhi.get("bhi_z")
                segments[i].bhi_pct = bhi.get("bhi_pct")
                segments[i].category = bhi.get("category")

            if i == l4_i:
                bar, bar_parts, trab_roi, l4_window = (
                    bar_i, bar_parts_i, trab_roi_i, crop_box_i
                )
                l4_qc_message = qc_message_i

        # 금속 자체는 신뢰성 있게 검출하지 못한다(밝기·형태 휴리스틱 모두 실측에서
        # 실패). _vertebra_qc()가 이미 'ROI가 포화됐다 = 값이 잘렸다' 등 객관적으로
        # 측정 가능한 것만으로 L4 QC를 판정했으므로, 그 결과를 study 신뢰도에 반영.
        qc_msgs: list[str] = [f"L4: {l4_qc_message}"] if l4_qc_message else []
        bar_fallback = bar is None
        # 밀도 히트맵/격자는 '원시 감쇠(dens)' 픽셀을 색칠한다. 따라서 스케일 앵커도
        # 반드시 같은 원시 도메인이어야 한다. BAR 단위(0~0.4)를 넣으면 raw 값(수천)이
        # 상한을 수천 배 초과해 ROI 전체가 최대색으로 포화된다(v17 초기 버그).
        # -> bmd_value만 BAR로 바꾸고, 시각화 스케일은 기존 원시 도메인을 유지한다.
        low_anchor, high_anchor = self._density_scale(
            dens, valid, l4_window, bone_dil, vals,
        )
        grid_src = dens
        if not bar_fallback:
            bmd_value = bar
        else:
            # BAR 실패(연부조직 기준값/soft-air 마진 불량) -> 레거시 0..1 점수로 폴백.
            # 스케일이 다르므로 추세에 섞이지 않도록 아래에서 경고를 남긴다.
            logger.warning(
                "BAR 산출 실패(margin=%s) -> 레거시 점수 폴백: %s",
                bar_parts.get("margin"), dicom_path.name,
            )
            bmd_value = self._bmd_score(mean_d, low_anchor, high_anchor)
        # 게이지 마커는 low_anchor~high_anchor(원시 도메인) 위의 위치여야 한다.
        # BAR(0.1 수준)을 여기 넣으면 스케일 맨 아래에 붙어 'Mean: 0'으로 표시된다.
        roi_marker = round(mean_d, 1)

        # ---- QC 판정을 신뢰도에 반영 ----
        # (_reliability()는 이 위에서 이미 실행됐으므로 여기서 결과만 덧붙인다)
        if bar_fallback:
            # 스케일이 다른 두 산식이 한 추세선에 섞이면 임상적으로 오해를 부른다.
            qc_msgs.append(
                "BAR 산출 실패로 레거시 0..1 점수 사용 — BAR 기반 값과 추세를 섞지 마세요."
            )
        if qc_msgs:
            # 값은 그대로 두되(임상의가 원본을 볼 수 있어야 한다) 신뢰도만 낮춘다.
            joined = " / ".join(qc_msgs)
            reliability_warning = (
                f"{reliability_warning} / {joined}" if reliability_warning else joined
            )
            reliable = False
            logger.info("QC 경고 (%s): %s", dicom_path.name, joined)
        # 종적 대조용 L4 밀도 격자 (post−pre로 골손실 부위 검출)
        density_grid = self._density_grid(
            grid_src, mask, valid, low_anchor, high_anchor, settings.bmd_diff_grid_n
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
        #   2) 모델 주목도: L4 크롭 Seg-Grad-CAM 어트리뷰션 (실제 그래디언트)
        xai_overlay_name = None   # 밀도 근거 히트맵 (faithful)
        xai_l4_cam_name = None    # Seg-Grad-CAM 크롭
        l4_box = self._crop_box(H, W, boxes[l4_i])
        try:
            xai_overlay_name = "xai_density.png"
            # l4_box(pad=0.25) — l4_crop.png와 동일한 프레이밍. l4_window는
            # pad=0.6짜리 넓은 창(soft-tissue 기준값 계산용)이라 여기 쓰면
            # 인접 척추까지 넓게 잡혀 "Extracted L4 vertebra"보다 훨씬 넓은
            # 영역이 보인다는 리포트로 발견 — 시각화 크롭은 l4_box로 통일.
            self._build_density_heatmap(
                rgb, bar_parts.get("bar_map"), trab_roi, l4_box, bar,
                str(output_dir / xai_overlay_name),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("밀도 히트맵 생성 실패 (생략): %s", exc)
            xai_overlay_name = None
        if grad_cam is not None:
            try:
                img_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                cam, _cam_masks = grad_cam.compute(img_bgr)
                if cam is not None:
                    cam_crop = self._build_l4_cam_crop(rgb, cam, l4_box)
                    xai_l4_cam_name = "xai_l4_cam.png"
                    cv2.imwrite(
                        str(output_dir / xai_l4_cam_name),
                        cv2.cvtColor(cam_crop, cv2.COLOR_RGB2BGR),
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Seg-Grad-CAM 크롭 생성 실패 (생략): %s", exc)
                xai_l4_cam_name = None

        #   3) L1~L5 전체 BAR 히트맵: 노트북 v18 E2와 동일한 5-패널 그림
        #      (패널마다 diverging/sequential 컬러맵 자동 전환 + 컬러바 + 흰 ROI 윤곽선).
        xai_bar_l1l5_name = None
        try:
            xai_bar_l1l5_name = "xai_bar_l1l5.png"
            self._build_bar_maps_l1l5(
                rgb, panels_by_label, str(output_dir / xai_bar_l1l5_name)
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("L1~L5 BAR 히트맵 생성 실패 (생략): %s", exc)
            xai_bar_l1l5_name = None
        xai_bar_l1l5_path = (
            f"{output_dir.name}/{xai_bar_l1l5_name}" if xai_bar_l1l5_name else None
        )

        #   4) 기준값 근거: air/연부조직 기준값이 실제로 어느 픽셀에서 왔는지 +
        #      BAR 계산식을 전체 이미지 위에 표시 (다른 패널과 달리 크롭 안 함).
        xai_air_soft_name = None
        try:
            xai_air_soft_name = "xai_air_soft.png"
            self._build_air_soft_heatmap(
                gray_lin, r.masks.xy, l4_i, valid, bone_dil, l4_window, view,
                bar, bar_parts, str(output_dir / xai_air_soft_name),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("air/연부조직 기준값 히트맵 생성 실패 (생략): %s", exc)
            xai_air_soft_name = None

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
        xai_air_soft_path = (
            f"{output_dir.name}/{xai_air_soft_name}" if xai_air_soft_name else None
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
            xai_bar_l1l5_path=xai_bar_l1l5_path,
            xai_air_soft_path=xai_air_soft_path,
            # 밀도 히트맵 스케일 (파랑 끝 / 빨강 끝 / L4 평균)
            density_low=round(low_anchor, 1),
            density_high=round(high_anchor, 1),
            roi_mean_attenuation=roi_marker,
            l4_density_grid=density_grid,
            reliable=reliable,
            reliability_warning=reliability_warning,
            segments=segments,
            xai_factors=xai,
            acquired_at=acquired_at,
            modality=modality,
            dicom_meta=dicom_meta,
            view_position=view,
            view_source=view_source,
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
