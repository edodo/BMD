"""L4 분할(YOLO) 진단 도구 — trabecular ROI가 실제로 어디서 만들어지는지,
침식/종판 절단 *이전*의 원본 폴리곤을 눈으로 확인한다.

배경
----
Density basis 패널의 흰 윤곽선(trabecular ROI)은 `_trabecular_roi()`가
YOLO 분할 폴리곤을 침식 + 상하 종판 절단해서 만든다. 즉 이 윤곽선은 YOLO가
그린 폴리곤 경계 '안쪽'만 깎아낼 수 있을 뿐, 폴리곤 자체가 척추체 경계를
벗어나 있으면(디스크 공간/인접 구조까지 포함) 침식을 아무리 늘려도 한계가
있다 — "해면골 추출이 정교하지 않다"는 리포트(특히 측면 사진 사례)를 확인
하기 위한 도구. 계산식/파라미터는 건드리지 않는다(진단 전용).

이 스크립트는 실제 YOLO 세그멘테이션 모델을 로드해서 돌리므로(디코딩만
하는 debug_air_reference.py와 달리) ultralytics/torch와 모델 가중치
(settings.ml_model_dir/settings.ml_model_file)가 있는 환경에서 실행해야
한다.

사용법 (backend 디렉터리에서, 모델/torch가 설치된 환경)
-----------------------------------------------------
    python scripts/debug_l4_segmentation.py <dicom_path> [out.png]

노란선 = YOLO가 그대로 내놓은 원본 L4 폴리곤 (침식/종판 절단 전)
흰선   = 실제 BAR 계산에 쓰이는 trabecular ROI (침식 + 종판 절단 후)
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

_BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from app.services.inference.yolo_engine import CONF, IMGSZ, YoloBmdEngine


def build_debug_image(dicom_path: Path) -> tuple[np.ndarray, dict]:
    engine = YoloBmdEngine()
    model = engine._ensure_model()

    rgb, dens, H, W, *_ = YoloBmdEngine._load_dicom(dicom_path)

    r = model.predict(rgb, conf=CONF, imgsz=IMGSZ, verbose=False)[0]
    if r.masks is None or len(r.boxes) == 0:
        raise RuntimeError("No vertebrae detected in this image")

    boxes = r.boxes.xyxy.cpu().numpy()
    classes = r.boxes.cls.cpu().numpy().astype(int)
    confs = r.boxes.conf.cpu().numpy()

    l4_i, status = YoloBmdEngine._pick_l4_index(boxes, classes, confs)
    if l4_i is None:
        raise RuntimeError(f"Could not identify L4: {status}")

    raw_poly = np.asarray(r.masks.xy[l4_i], dtype=np.float32)  # (N,2) 원본 좌표
    full_mask = YoloBmdEngine._poly_to_mask(raw_poly, H, W, shrink=1.0)
    trab_roi = YoloBmdEngine._trabecular_roi(full_mask)

    # xai_density.png / l4_crop.png와 동일한 프레이밍(pad=0.25)으로 크롭.
    l4_box = YoloBmdEngine._crop_box(H, W, boxes[l4_i])
    cx1, cy1, cx2, cy2 = l4_box

    crop_rgb = rgb[cy1:cy2, cx1:cx2].copy()
    crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)

    # 노란선: YOLO가 내놓은 원본 폴리곤 (침식/종판 절단 전)
    poly_local = (raw_poly - np.array([cx1, cy1], dtype=np.float32)).astype(np.int32)
    cv2.polylines(crop_bgr, [poly_local], True, (0, 220, 255), 2, cv2.LINE_AA)

    # 흰선: 실제 BAR 계산에 쓰이는 trabecular ROI (침식 + 종판 절단 후)
    trab_crop = trab_roi[cy1:cy2, cx1:cx2]
    contours, _ = cv2.findContours(
        (trab_crop > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(crop_bgr, contours, -1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.rectangle(crop_bgr, (0, 0), (crop_bgr.shape[1], 44), (0, 0, 0), -1)
    cv2.putText(crop_bgr, "yellow = raw YOLO L4 polygon (pre-erosion)", (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(crop_bgr, "white = trabecular ROI actually used for BAR", (8, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    ch, cw = crop_bgr.shape[:2]
    if 0 < cw < 480:
        scale = 480 / cw
        crop_bgr = cv2.resize(
            crop_bgr, (int(cw * scale), int(ch * scale)), interpolation=cv2.INTER_CUBIC
        )

    poly_area = float(cv2.contourArea(poly_local.astype(np.float32)))
    info = {
        "status": status,
        "l4_conf": float(confs[l4_i]),
        "poly_vertices": int(len(raw_poly)),
        "poly_area_px": poly_area,
        "full_mask_px": int(full_mask.sum()),
        "trab_roi_px": int(trab_roi.sum()),
    }
    return crop_bgr, info


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python scripts/debug_l4_segmentation.py <dicom_path> [out.png]")
        raise SystemExit(1)

    dicom_path = Path(sys.argv[1])
    out_path = (
        Path(sys.argv[2]) if len(sys.argv) > 2
        else dicom_path.with_suffix(".l4_seg_debug.png")
    )

    img, info = build_debug_image(dicom_path)
    cv2.imwrite(str(out_path), img)

    print(f"[{dicom_path.name}]")
    print(f"  L4 선택 방식: {info['status']} (confidence {info['l4_conf']:.3f})")
    print(f"  원본 폴리곤: 꼭짓점 {info['poly_vertices']}개, 면적 {info['poly_area_px']:.0f}px")
    print(f"  전체 마스크: {info['full_mask_px']}px -> 침식+종판절단 후 trabecular ROI: "
          f"{info['trab_roi_px']}px "
          f"({info['trab_roi_px'] / max(info['full_mask_px'], 1) * 100:.1f}% 유지)")
    print(f"  저장: {out_path}")


if __name__ == "__main__":
    main()
