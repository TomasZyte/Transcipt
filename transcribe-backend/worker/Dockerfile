# GPU-воркер: ffmpeg + WhisperX (ASR/выравнивание/диаризация) + клиент LLM.
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git && rm -rf /var/lib/apt/lists/*

# CTranslate2 (внутри faster-whisper/WhisperX) требует cuDNN 8, а базовый образ несёт cuDNN 9.
# Ставим cuDNN 8 в отдельную папку и добавляем в путь, НЕ трогая cuDNN 9, которым пользуется torch.
RUN pip install --no-cache-dir --target=/opt/cudnn8 nvidia-cudnn-cu12==8.9.7.29
ENV LD_LIBRARY_PATH=/opt/cudnn8/nvidia/cudnn/lib:${LD_LIBRARY_PATH}

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Модели вшиваем в образ: сетевой том привязал бы эндпоинт к одному
# датацентру, а A5000 и 3090 живут в разных.
# Этот слой стоит ДО копирования кода — иначе правка pipeline.py
# заставляла бы качать все модели заново.
ENV HF_HOME=/models \
    HF_HUB_ENABLE_HF_TRANSFER=0

ARG ASR_REPO=Systran/faster-whisper-large-v3
ARG LLM_REPO=Qwen/Qwen2.5-3B-Instruct
ARG ALIGN_RU=jonatasgrosman/wav2vec2-large-xlsr-53-russian

RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('${ASR_REPO}'); \
snapshot_download('${LLM_REPO}'); \
snapshot_download('${ALIGN_RU}')"

# Код меняется часто — держим его последним слоем
COPY pipeline.py handler.py local_server.py ./

CMD ["python", "handler.py"]

