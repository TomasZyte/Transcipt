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

COPY pipeline.py handler.py local_server.py ./

CMD ["python", "handler.py"]