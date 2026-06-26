# 백엔드 요구사항 정의서 — BMD Viewer

문서 버전: 1.0
작성일: 2026-06-25

---

## 1. 개요
백엔드는 FastAPI와 SQLite(비동기 SQLAlchemy)로 구현하며, 인증, 환자/스터디 관리, DICOM 업로드, 비동기 추론, 골밀도 추세 산출을 담당한다. 추론 엔진은 인터페이스로 분리하여 기존 노트북 파이프라인을 이식할 수 있게 하고, 다른 프로젝트에 통합 가능하도록 모듈 경계를 명확히 한다.

## 2. 기술 요구사항
런타임은 Python 3.12이며, 주요 의존성은 FastAPI, SQLAlchemy(async), aiosqlite, python-jose(JWT), bcrypt이다. 데이터베이스는 SQLite를 기본으로 하되 연결 URL 교체만으로 PostgreSQL 등으로 전환 가능해야 한다. 추론 시 GPU(CUDA)와 CPU를 환경변수로 선택할 수 있어야 한다.

## 3. 아키텍처 및 모듈 구조

백엔드는 계층별로 분리된다. API 계층(`api/`)은 엔드포인트와 의존성을 담고, 핵심 설정/보안(`core/`), 데이터베이스 세션(`db/`), ORM 모델(`models/`), 입출력 스키마(`schemas/`), 서비스 계층(`services/`), 비동기 워커(`workers/`)로 구성된다.

추론 로직은 `services/inference/engine.py`의 `BmdInferenceEngine` 추상 클래스를 통해 분리되며, 노트북 파이프라인은 이 클래스의 구현으로 이식한다. 애플리케이션은 `create_app()` 팩토리를 제공하여 상위 애플리케이션에 마운트하거나 라우터를 통합할 수 있다.

## 4. 데이터 모델
주요 테이블은 doctors, patients, xray_studies, bmd_measurements, vertebra_segments, xai_factors이다. 환자는 의사에 소속되며, 스터디는 환자에 소속된다. 측정 결과는 스터디와 1:1 관계이며, 분할과 XAI 항목을 자식으로 가진다. 골밀도 기준 척추는 L4이며 분할의 is_target 플래그로 식별한다.

## 5. API 요구사항

### 5.1 인증
| 메서드 | 경로 | 설명 |
|--------|------|------|
| POST | /api/v1/auth/register | 의사 등록(이메일, 비밀번호, 이름, 역할) |
| POST | /api/v1/auth/login | 로그인, JWT 액세스 토큰 발급 |

비밀번호는 bcrypt로 해싱하며 72바이트를 초과하면 안전하게 절단한다. 토큰은 사용자 식별자와 역할을 포함하고 만료시간은 환경변수로 설정한다.

### 5.2 환자
| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | /api/v1/patients | 목록 조회 + 이름/MRN 검색(q), 페이지네이션 |
| POST | /api/v1/patients | 환자 등록 |
| GET | /api/v1/patients/{id} | 환자 상세 |
| PATCH | /api/v1/patients/{id} | 환자 정보 수정 |

목록 응답에는 각 환자의 누적 스터디 수와 최근 골밀도 값을 포함한다. 의사는 자신이 담당하는 환자만 조회할 수 있다.

### 5.3 X-ray 스터디
| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | /api/v1/patients/{id}/studies | 환자의 X-ray 이력(최신순) |
| POST | /api/v1/patients/{id}/studies | DICOM 업로드, 202 반환 후 비동기 추론 |
| GET | /api/v1/studies/{id} | 스터디 상세(원본 + 분할 + L4 BMD + XAI) |

업로드는 DICOM 파일을 저장하고 스터디를 생성한 뒤 즉시 202(Accepted)를 반환한다. 추론은 백그라운드 작업으로 실행된다.

### 5.4 골밀도 추세
| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | /api/v1/patients/{id}/bmd-trend | L4 골밀도 시계열과 변화량 요약 |

응답에는 측정 시점별 골밀도 포인트와, 최초 대비 최신의 절대 변화량 및 백분율 변화량을 포함한다.

## 6. 비동기 추론 처리
업로드 직후 추론은 백그라운드에서 실행된다. 처리 흐름은 스터디 상태를 PROCESSING으로 전환, 추론 엔진 실행(DICOM 로드 및 전처리, 척추 검출, L4 분할, proxy BMD 점수화, Grad-CAM 설명성, XAI 항목 산출), 결과 저장, 상태를 COMPLETED로 전환하는 순서다. 예외 발생 시 상태를 FAILED로 전환하고 사유를 기록한다. 부하 증가 시 동일 인터페이스로 큐 기반(Celery 등)으로 전환할 수 있다.

## 7. 추론 엔진 이식 요구사항
노트북의 척추체 추출 및 골밀도 계산 로직은 `BmdInferenceEngine.run(dicom_path, output_dir)` 메서드를 구현하여 이식한다. 반환 객체(InferenceResult)는 골밀도 값, 기준 척추(L4), T-score, 신뢰도, 노출 보정 여부, 미리보기/Grad-CAM 경로, 척추 분할 목록, XAI 기여 항목 목록을 포함한다. 이식 전까지는 StubBmdEngine으로 배선을 검증한다.

## 8. 스토리지
원본 DICOM은 storage/dicom 아래에, 추론 파생 산출물(미리보기 PNG, 분할 마스크, Grad-CAM 오버레이)은 storage/derived 아래에 저장한다. 파생 산출물은 정적 경로(/derived)로 서빙한다. 컨테이너 배포 시 스토리지는 볼륨으로 분리한다.

## 9. 보안 요구사항
모든 환자/스터디 API는 인증을 요구한다. 의사는 자신의 환자 데이터에만 접근 가능하다. 역할(admin/doctor/radiologist/viewer) 기반 접근 제어를 위한 의존성을 제공하여 향후 권한 분리를 지원한다. 비밀키는 운영 환경에서 환경변수로 주입한다.

## 10. 배포
Dockerfile은 Python 3.12-slim 기반으로 의존성 설치 후 uvicorn으로 실행한다. 무거운 추론은 필요 시 별도 워커 컨테이너로 분리할 수 있다. 데이터베이스 스키마는 개발 단계에서 자동 생성하되, 운영에서는 마이그레이션 도구(Alembic) 사용을 권장한다.
