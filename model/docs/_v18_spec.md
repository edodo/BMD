# v18 Spec — 요추 L1–L5 골조직 시계열 분석 파이프라인

> **이 문서는 Claude Code 세션용 사양서입니다.**
> v17 측정 코어 계승 · L1–L5 일반화 · 무라벨 AE(이상탐지+조기경보) · SHAP · 시계열

---

## 0. 사용법 (Claude Code에게)

세션 시작 시 순서:

1. 이 문서(`v18_spec.md`) 전체를 읽는다.
2. 기존 `L4_AP_segmentation_v17.ipynb`를 읽는다. **Section "계승 원칙"의 함수들은 이미
   검증되었으므로 재발명하지 말고 계승·리팩터한다.**
3. 아래 **진행 규칙**을 따른다 — 한 번에 A~F 전체를 생성하지 말 것.
4. 모든 문서는 영어로 작성한다.
5. L4_AP_segmentation_v18.ipynb 파일을 생성한다.

### 진행 규칙 (중요)
- **섹션 단위로 진행하고, 각 섹션 끝에서 검증 게이트를 통과한 뒤 다음으로 넘어간다.**
- 권장 순서: **B(측정 코어) → 검증 → C(AE) → D(시계열) → E(XAI) → A/F 마감.**
  - B를 먼저 하는 이유: 측정 코어가 L1–L5에서 정확히 돌고 기존 L4 수치와 일치함을
    확인해야, 이후 오류가 났을 때 격리가 된다.
- 각 단계에서 "왜 이렇게 설계했는가"를 수식·물리적 근거와 함께 주석/마크다운으로 남긴다.
- 결과가 불확실하면 임의로 진행하지 말고 사용자에게 확인을 요청한다.

### 환경 (이미 설정 완료 — 참고용)
- Windows / RTX 5060 (Blackwell, sm_120) / PyTorch nightly cu128 — **설정 완료됨.**
  재설치·환경 변경 시도 금지. GPU 인식 확인 셀만 포함.
- 모델 가중치(절대경로, 재학습 금지):
  `C:\Users\csm02\Desktop\edward\bmd\src\BMD\model\runs\segment\ap_seg\l1_l5_yolo26m_ap_la\weights\best.pt`

---

## 1. 역할

너는 Medical Imaging AI 시니어 엔지니어다. 연구 스크립트가 아니라 유지보수 가능한
프로젝트 수준으로, 함수는 단일 책임·재사용 가능하게 모듈화하고, 각 설계 결정에
수식·물리적 근거·"왜 이렇게 했는가"를 주석으로 남긴다.

---

## 2. 두 종류의 학습을 명확히 구분 (혼동 금지)

- **YOLO26-seg**: 이미 학습 완료. 새 파이프라인에서는 **추론(inference) 전용, 재학습 절대
  없음.** 역할은 "L1–L5가 어디 있나"까지. 골밀도 변화는 판단하지 않는다.
- **오토인코더(AE)**: 완전히 별개의 새 모델(YOLO와 무관). **라벨 없이** 학습.
  역할은 아래 두 가지로 명확히 규정된다 — **장식용 학습 금지**:
  1. **다변량 이상탐지** — BAR·GLCM·fractal·variogram 각각은 정상 범위여도 그 *조합*이
     비정상일 수 있다. AE는 정상 척추 특징의 결합 분포를 학습하므로, 재구성오차가 크면
     "개별 지표는 정상인데 패턴이 이상"을 잡는다. 단순 임계값으로는 불가능한 기여.
  2. **시계열 조기경보** — 동일 환자 latent 궤적의 이동 속도/방향이 BAR 절대값보다 먼저
     변화를 감지. LSC와 결합해 "BAR는 아직 LSC 미달이나 latent가 이미 이동 중" 소견 생성.

---

## 3. 계승 원칙 (v17에서 이미 검증됨 — 재발명 금지, 계승·정교화, L4→L1–L5 일반화)

`v17` 노트북의 아래 방법론은 이미 옳고 검증되었다. 재구현하지 말고 함수로 계승하되,
L4 하드코딩을 **L1–L5 일반형(척추 루프)**으로 리팩터한다.

- **BAR** = (A_trab − A_soft)/(A_soft − A_air): 노출·게인·window/level 불변
  (affine A→aA+b 에서 offset b·gain a 소거). **절대 강도 μ를 지표로 직접 쓰지 않는다.**
- **to_attenuation()**: DICOM `PixelIntensityRelationship`
  (LIN→A=−ln(1−I), LOG→A=I).
- **air_reference()**: Otsu 배경 분리 후 25퍼센타일(직접노출 배경).
- **soft_tissue_ref()**: 척추 측방 paraspinal 밴드. 일측성 오염(posterior element/rib)
  대응 위해 median이 아닌 **저차 퍼센타일 추정기**. 인접 척추 `exclude` 마스크로 누출 차단.
- **trabecular_roi()**: 피질rim erosion + endplate 밴드 제거 → 해면골 ROI.
  (주의: 2D 투사에서 피질/해면 *분리*가 아니라 공간적 *제외*임을 명시.)
- **QC 게이트**: soft-air margin 미달 시 BAR=None·FAIL, "**FAIL은 해석하지 않음**".
- **텍스처**: GLCM(대비·상관·에너지·균질성·엔트로피), fractal dimension(box-counting),
  variogram slope(TBS-like). 각 특징의 healthy 방향(sign)은 **코호트에서 학습, 가정 금지**.
- **LSC**: RMS-CV% × 2.77 (ISCD, 95%). 이보다 작은 변화는 측정 노이즈.
- **Seg-Grad-CAM**: 추론캐시가 아닌 **별도 autograd 모델 인스턴스**, L1..L5 mask score
  합에 단일 backward. per-pixel BAR heatmap은 스칼라 BAR와 수치적으로 일치(검증 출력).

---

## 4. 절대 준수 (v17 한계 문서에서 확정된 정직성 원칙)

1. **areal BMD(g/cm²) 산출 금지** — 보정 팬텀 없음. BAR는 질량이 아닌 비율.
2. **T-score 아님** — 코호트 z-score. "osteoporosis-like"이지 osteoporosis 아님.
3. **DXA 미검증** — BHI 가중치·컷포인트·텍스처 polarity·AE 이상임계값 모두 라벨 도착 전까지
   placeholder. CSV는 라벨 붙으면 재적합 가능하도록 형상 유지.
4. **지도 회귀·RBDI-MSE 손실 금지** — 라벨 없음. AE 재구성 손실만 사용.

---

## 5. 사용 모델

- **세그멘테이션(추론 전용)**: YOLO26-seg 사용자 가중치 (위 절대경로), 클래스 L1..L5.
- **무라벨 표현학습(학습 대상)**: 특징 오토인코더(AE).
  - 입력 = 척추별 특징벡터(BAR + GLCM 5종 + fractal + variogram, 표준화).
  - encoder → latent → decoder. 재구성오차 = 이상점수. latent 궤적 = 조기경보 신호.

---

## 6. 노트북/모듈 구조 (섹션별 항목 분류)

### A. 기반 (Foundation)
- **A1** 환경설정 — seed 고정, 출력트리, `config.yaml` 중앙화, GPU 인식 확인
- **A2** 데이터 수집 — DICOM 태그(patient_id, study_date, view, kVp/mAs/SID,
  PixelSpacing, PixelIntensityRelationship) 파싱 → DataFrame, 환자·시간순 정렬
- **A3** 품질 필터링 — Laplacian variance 블러 검출, L1–L5 가시성 체크
- **A4** 전처리 — rescale, MONOCHROME 반전, robust windowing, (선택)CLAHE
- **A5** 데이터 분할 — patient-wise(train/val), 시계열 test 보존

### B. 측정 코어 (Measurement Core — v17 계승·L1–L5 일반화)
- **B1** 참조 & ROI — air/soft/trabecular (L1–L5 루프)
- **B2** 금속 임플란트 처리 — 고강도 포화 검출 → metal mask, ROI 제외/플래그(골반수술 필수)
- **B3** BAR 산출 — 척추별 반복, to_attenuation 도메인
- **B4** 텍스처 — GLCM/fractal/variogram, 척추별
- **B5** BAR 노출불변 수치증명 — affine 전후 BAR 불변 vs 구식 ratio 드리프트
- **B6** per-pixel BAR map — 스칼라 BAR와 일치 검증
- ▸ 시각화: L1–L5 마스크 오버레이(척추별 컬러), 참조밴드, 척추별 BAR 막대

> **B 검증 게이트**: L1–L5 각 척추에서 BAR/텍스처가 계산되고 QC가 동작하는가?
> 기존 v17의 L4 수치와 새 파이프라인의 L4 수치가 **일치**하는가? (동일 이미지 대조)
> 통과 후 C로 진행.

### C. 무라벨 AE (이상탐지 + 조기경보)
- **C1** 특징행렬 — 척추×특징, 표준화, 정상 코호트로 fit
- **C2** AE 설계 — encoder/decoder 대칭 구조 다이어그램
- **C3** 활성함수 비교 — ReLU/GELU/SiLU, 출력 linear, 곡선 + 근거
- **C4** 손실 — 재구성 MSE(+선택 L1/KL), 형태 비교. (지도 RBDI 손실 미사용 명시)
- **C5** 옵티마이저 — AdamW + CosineAnnealing(WarmRestart), LR 스케줄 곡선
- **C6** 반복학습 — epoch/early stopping/checkpoint, train/val 손실·gradient flow
- **C7** 검증(무라벨) — 재구성오차 분포, latent PCA/UMAP, 반복촬영 일관성(smoothness)
- **C8** HPO — Optuna(latent차원/lr/dropout, 목표 val 재구성손실), history·importance
- **C9** 이상점수 정의 — 재구성오차 → 척추별 anomaly score, 코호트 백분위

> **C 검증 게이트**: AE가 수렴하는가(train/val 손실 하강)? 재구성오차가 정상 척추에서
> 낮고 명백한 이상 케이스에서 높은가(sanity)? 통과 후 D로 진행.

### D. 시계열 (Temporal — 프로젝트 본래 목적)
- **D1** 시계열 조립 — 환자별 study_date순 척추별 BAR·BHI·anomaly·latent 트랙
  - ▸ **실데이터 있으면 실측, 없으면 합성 다중visit로 데모**(코드 동일, 데이터소스만 분기)
- **D2** 변화 유의성 — ΔBAR(t) vs LSC(노이즈 vs 실변화) + 조기경보(latent 이동이 LSC
  도달보다 선행하는지 판정)
- **D3** 시각화 — 척추별 시계열 라인(L1–L5 서브플롯), ΔBAR 히트맵, spider chart,
  latent 궤적

### E. 설명가능성 (Explainability)
- **E1** 세그멘테이션 XAI — Seg-Grad-CAM(L1..L5, 실제 gradient) + (선택)LayerCAM
- **E2** 측정 XAI — per-pixel BAR 기여 heatmap, 척추별
- **E3** SHAP — AE 이상점수 대상 KernelSHAP(또는 DeepSHAP). 어떤 특징 조합이 이상을
  유발했는지 → summary(beeswarm)·waterfall·force·dependence
- ▸ 자연어 소견 생성 예:
  *"L3 variogram↑·fractal↓ 조합으로 AE 이상점수 상승. BAR는 아직 LSC 미달이나 latent
  이동이 선행 — 해면골 구조 이질성 증가 시사."*

### F. 산출물 (Deliverables)
- **F1** 척추별·환자별 clinical report card + report.json(schema 유지)
- **F2** 코호트 대시보드
- **F3** ONNX export(PyTorch-ONNX parity check)
- **F4** disclaimer: "상대 추세·코호트 z-score·AE 이상점수는 라벨없는 대리지표.
  절대 BMD/T-score 아님"

---

## 7. 정직한 한계 (리포트에 명시)

- areal BMD 아님 / T-score 아님 / DXA 미검증(placeholder) / LSC는 하한 /
  BAR는 AP, LA 모두 검증 / AE 이상임계값은 정상 라벨 확보 전까지 placeholder

---

## 8. 다음 단계 (라벨 도착 시)

DXA 라벨 ≥50 study 확보 시:
`corr(BAR, T-score)` · `ROC(BHI → osteoporosis)` · `ROC(AE anomaly → 진단)` 실행 →
`fit_reference` · `BHI_WEIGHTS` · AE 이상임계값 재적합.
모든 "-like"를 방어 가능한 임상 주장으로 전환.
