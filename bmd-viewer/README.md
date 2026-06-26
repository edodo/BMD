# BMD Viewer

2D X-ray(DICOM) 기반 L4 골밀도(BMD) 측정·추적 뷰어. 의사용.

## 빠른 시작

### 로컬 개발
```bash
# 백엔드
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload   # http://localhost:8000/docs

# 프론트엔드 (새 터미널)
cd frontend
npm install
npm run dev                     # http://localhost:5173
```

### Docker (전체)
```bash
docker compose up --build       # http://localhost
```

## 구조
- `backend/` — FastAPI + SQLite. 추론 엔진은 `app/services/inference/`에서 교체.
- `frontend/` — React+Vite. 데이터 계층은 `src/lib/data/`에서 백엔드 전환.
- `docs/` — 요구사항 정의서.

## 추론 파이프라인
`backend/app/services/inference/yolo_engine.py`의 `YoloBmdEngine`이
노트북(L4_AP_segmentation.ipynb) 파이프라인을 이식한 실제 구현입니다.
YOLOv8-seg(`ml_models/l1_l5_seg.pt`)로 L1~L5를 분할하고, L4 중앙
해면골 ROI의 강도 통계로 proxy BMD를 산출합니다.

GPU(RTX 5060) 사용 시 `BMD_INFERENCE_DEVICE=cuda`와 CUDA용 torch를 설치하세요.
모델 없이 배선만 테스트하려면 `BMD_INFERENCE_ENGINE=stub`.
