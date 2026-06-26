# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 시스템 의존성 (DICOM/이미지 처리 대비 최소)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app

# 학습된 모델 가중치 포함 (l1_l5_seg.pt).
# 대용량 모델은 빌드에 포함하는 대신 볼륨/오브젝트스토리지 마운트도 가능.
COPY ml_models ./ml_models

# 스토리지 디렉토리 (볼륨 마운트 권장)
RUN mkdir -p storage/dicom storage/derived

EXPOSE 8000

# 운영 시 --workers 조정. 비동기 추론이 무거우면 별도 워커 컨테이너 권장.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
