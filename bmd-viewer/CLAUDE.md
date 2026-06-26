# BMD Viewer — 프로젝트 지침 (Claude Code용)

## 개요
2D X-ray(DICOM)에서 L4 척추 골밀도(BMD)를 측정·추적하는 의사용 뷰어.
사용자=의사, 대상=환자. 비동기 추론으로 분할+BMD+XAI를 산출한다.

## 스택
- 백엔드: FastAPI + SQLAlchemy(async) + SQLite. 추론은 BackgroundTasks.
- 프론트: React + Vite + TypeScript + React Query + Recharts.
- 배포: Docker (backend/frontend) + docker-compose.

## 핵심 설계 원칙
1. **추론 엔진은 인터페이스로 분리** (app/services/inference/engine.py).
   실제 구현은 yolo_engine.py(YoloBmdEngine) — YOLOv8-seg로 L1~L5 분할 후
   L4 ROI의 dens 통계로 proxy BMD 산출. 모델: ml_models/l1_l5_seg.pt.
   BMD_INFERENCE_ENGINE=stub 으로 스텁 전환 가능(모델 없으면 자동 폴백).
2. **프론트 데이터 계층 추상화** (src/lib/data/provider.ts).
   SQLite(REST)→Firebase 전환 시 구현만 교체, 컴포넌트 불변.
3. **모듈화**: features/ 단위로 분리. 다른 프로젝트에 feature 폴더째 이식 가능.
4. DB 교체 대비: database_url만 바꾸면 PostgreSQL 등으로 전환.

## 명령
- 백엔드: `cd backend && uvicorn app.main:app --reload`
- 프론트: `cd frontend && npm run dev`
- 전체: `docker compose up --build`

## 주의
- 비밀번호 해싱은 bcrypt 직접 사용(72바이트 절단). passlib 미사용.
- L4가 BMD 기준 척추(target_vertebra). 분할 결과의 is_target 플래그로 식별.
