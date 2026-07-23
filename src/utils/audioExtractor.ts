export interface AudioChunk {
  base64: string;
  mimeType: string;
  offsetSec: number;
  durationSec: number;
}

export interface ProcessedAudioResult {
  durationSec: number;
  chunks: AudioChunk[];
  singleBase64?: string;
  singleMimeType?: string;
}

export async function processMediaFileForTranscription(
  file: File,
  onProgress?: (msg: string) => void
): Promise<ProcessedAudioResult> {
  const isVideo = file.type.startsWith('video/') || file.name.match(/\.(mp4|webm|mov|avi|mkv)$/i) !== null;

  if (onProgress) onProgress('Извлечение и подготовка аудиосигнала...');

  // Try browser AudioContext audio track extraction (works for both audio & video files)
  try {
    const arrayBuffer = await file.arrayBuffer();
    const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 16000 });

    const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
    const durationSec = audioBuffer.duration;

    if (onProgress) onProgress(`Аудиопоток декодирован (${Math.round(durationSec)} сек). Сжатие...`);

    // If audio is under 5 minutes (300 seconds), produce a single 16kHz mono WAV chunk
    const CHUNK_DURATION = 300; // 5 minutes per chunk (~9.6 MB mono WAV)
    const totalChunksCount = Math.ceil(durationSec / CHUNK_DURATION);

    if (totalChunksCount <= 1) {
      const slicedBuffer = sliceAudioBuffer(audioBuffer, 0, durationSec, audioCtx);
      const wavBlob = audioBufferToMonoWav(slicedBuffer, 16000);
      const base64 = await blobToBase64(wavBlob);
      await audioCtx.close();

      return {
        durationSec,
        chunks: [{
          base64,
          mimeType: 'audio/wav',
          offsetSec: 0,
          durationSec,
        }],
      };
    }

    // For longer files (>5 minutes), slice into 5-minute WAV chunks
    const chunks: AudioChunk[] = [];
    for (let i = 0; i < totalChunksCount; i++) {
      const startSec = i * CHUNK_DURATION;
      const endSec = Math.min((i + 1) * CHUNK_DURATION, durationSec);
      const chunkDuration = endSec - startSec;

      if (onProgress) {
        onProgress(`Разделение на части: Фрагмент ${i + 1} из ${totalChunksCount}...`);
      }

      const slicedBuffer = sliceAudioBuffer(audioBuffer, startSec, endSec, audioCtx);
      const wavBlob = audioBufferToMonoWav(slicedBuffer, 16000);
      const chunkBase64 = await blobToBase64(wavBlob);

      chunks.push({
        base64: chunkBase64,
        mimeType: 'audio/wav',
        offsetSec: startSec,
        durationSec: chunkDuration,
      });
    }

    await audioCtx.close();

    return {
      durationSec,
      chunks,
    };
  } catch (err) {
    console.warn('AudioContext decoding failed, fallback to direct media file base64:', err);

    if (onProgress) onProgress('Чтение медиафайла напрямую...');
    const base64 = await fileToBase64(file);

    return {
      durationSec: 0,
      chunks: [],
      singleBase64: base64,
      singleMimeType: file.type || (isVideo ? 'video/mp4' : 'audio/mp3'),
    };
  }
}

function sliceAudioBuffer(
  buffer: AudioBuffer,
  startSec: number,
  endSec: number,
  audioCtx: AudioContext
): AudioBuffer {
  const sampleRate = buffer.sampleRate;
  const startSample = Math.floor(startSec * sampleRate);
  const endSample = Math.min(Math.floor(endSec * sampleRate), buffer.length);
  const frameCount = Math.max(1, endSample - startSample);

  const slicedBuffer = audioCtx.createBuffer(1, frameCount, sampleRate);
  const channelData = slicedBuffer.getChannelData(0);
  const originalLeft = buffer.getChannelData(0);

  if (buffer.numberOfChannels > 1) {
    const originalRight = buffer.getChannelData(1);
    for (let i = 0; i < frameCount; i++) {
      const idx = startSample + i;
      channelData[i] = ((originalLeft[idx] || 0) + (originalRight[idx] || 0)) / 2;
    }
  } else {
    for (let i = 0; i < frameCount; i++) {
      channelData[i] = originalLeft[startSample + i] || 0;
    }
  }

  return slicedBuffer;
}

function fileToBase64(file: File | Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

function blobToBase64(blob: Blob): Promise<string> {
  return fileToBase64(blob);
}

function audioBufferToMonoWav(buffer: AudioBuffer, targetSampleRate = 16000): Blob {
  const numberOfChannels = buffer.numberOfChannels;
  const length = Math.floor(buffer.duration * targetSampleRate);

  const monoData = new Float32Array(length);
  const left = buffer.getChannelData(0);
  const ratio = buffer.sampleRate / targetSampleRate;

  for (let i = 0; i < length; i++) {
    const originalIndex = Math.floor(i * ratio);
    if (originalIndex < left.length) {
      if (numberOfChannels > 1) {
        let sum = 0;
        for (let c = 0; c < numberOfChannels; c++) {
          sum += buffer.getChannelData(c)[originalIndex] || 0;
        }
        monoData[i] = sum / numberOfChannels;
      } else {
        monoData[i] = left[originalIndex];
      }
    }
  }

  const wavBuffer = createWavBuffer(monoData, targetSampleRate);
  return new Blob([wavBuffer], { type: 'audio/wav' });
}

function createWavBuffer(samples: Float32Array, sampleRate: number): ArrayBuffer {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  writeString(view, 0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(view, 8, 'WAVE');
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(view, 36, 'data');
  view.setUint32(40, samples.length * 2, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i++, offset += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }

  return buffer;
}

function writeString(view: DataView, offset: number, string: string) {
  for (let i = 0; i < string.length; i++) {
    view.setUint8(offset + i, string.charCodeAt(i));
  }
}
