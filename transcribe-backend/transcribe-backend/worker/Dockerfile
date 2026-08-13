# GPU-воркер: ffmpeg + WhisperX (ASR/выравнивание/диаризация) + клиент LLM.
# База с уже установленными torch+CUDA+cudnn — это снимает 90% проблем сборки.
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pipeline.py handler.py local_server.py ./

# (опционально) прогреть веса в образ ради быстрого холодного старта —
# иначе они качаются с HuggingFace при первом старте. Задайте HF_TOKEN при сборке:
#   docker build --build-arg HF_TOKEN=hf_xxx -t <user>/transcribe-worker .
# ARG HF_TOKEN
# ENV HF_TOKEN=$HF_TOKEN
# RUN python -c "import whisperx; whisperx.load_model('large-v3','cpu',compute_type='int8')"

# RunPod Serverless — точка входа по умолчанию:
CMD ["python", "handler.py"]

# Локальный dev-воркер (docker-compose.dev) переопределяет CMD на:
#   uvicorn local_server:app --host 0.0.0.0 --port 8001
