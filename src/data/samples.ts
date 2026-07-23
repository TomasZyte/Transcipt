import { AudioSample } from '../types';

export const DEMO_SAMPLES: AudioSample[] = [
  {
    id: 'sample-meeting',
    title: 'Планирование спринта и обсуждение релиза',
    category: 'Рабочая встреча',
    duration: '00:45',
    description: 'Обсуждение нового функционала мобильного приложения, сроков и распределения задач между разработчиками.',
    mimeType: 'audio/mp3',
    // We can generate synthetic or sample sound using WebAudio or prebuilt data
  },
  {
    id: 'sample-interview',
    title: 'Интервью с Senior Frontend Разработчиком',
    category: 'Собеседование',
    duration: '01:10',
    description: 'Вопросы по архитектуре React, оптимизации рендеринга и опыту работы с TypeScript и Gemini API.',
    mimeType: 'audio/mp3',
  },
  {
    id: 'sample-lecture',
    title: 'Мини-лекция: Искусственный интеллект в 2026 году',
    category: 'Образование',
    duration: '00:55',
    description: 'Обзор генеративных моделей, мультимодальных агентов и их применения в автоматизации бизнеса.',
    mimeType: 'audio/mp3',
  },
  {
    id: 'sample-podcast',
    title: 'Подкаст: Как запустить стартап в IT',
    category: 'Подкаст',
    duration: '01:05',
    description: 'Разбор ошибок начинающих фаундер-разработчиков, поиск инвесторов и валидация продуктовых гипотез.',
    mimeType: 'audio/mp3',
  }
];

/**
 * Generates a realistic audio Blob (synthesized audio with voice/tone)
 * for testing when a sample is selected.
 */
export async function generateSampleAudioBlob(sampleId: string): Promise<{ blob: Blob; mimeType: string }> {
  // Use Web Audio API to generate a multi-tone harmonic sound clip or speech synthesis if supported
  const ctx = new (window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext)({ sampleRate: 22050 });
  const durationSec = sampleId === 'sample-interview' ? 12 : sampleId === 'sample-lecture' ? 10 : 8;
  const sampleRate = ctx.sampleRate;
  const frameCount = sampleRate * durationSec;
  const myArrayBuffer = ctx.createBuffer(1, frameCount, sampleRate);
  const nowBuffering = myArrayBuffer.getChannelData(0);

  // Generate pleasant harmonic speech-like audio modulation
  for (let i = 0; i < frameCount; i++) {
    const t = i / sampleRate;
    // Simulate vocal cadence with formants and envelope
    const cadence = Math.sin(2 * Math.PI * 3 * t) * 0.5 + 0.5;
    const tone1 = Math.sin(2 * Math.PI * 220 * t);
    const tone2 = Math.sin(2 * Math.PI * 440 * t) * 0.5;
    const tone3 = Math.sin(2 * Math.PI * 880 * t) * 0.25;
    const voiceEnvelope = (tone1 + tone2 + tone3) * cadence * 0.2;
    // add small breath noise
    const noise = (Math.random() * 2 - 1) * 0.02 * cadence;
    nowBuffering[i] = voiceEnvelope + noise;
  }

  // Convert AudioBuffer to WAV blob
  const wavBlob = audioBufferToWavBlob(myArrayBuffer);
  ctx.close();
  return { blob: wavBlob, mimeType: 'audio/wav' };
}

function audioBufferToWavBlob(buffer: AudioBuffer): Blob {
  const numOfChan = buffer.numberOfChannels;
  const length = buffer.length * numOfChan * 2 + 44;
  const out = new DataView(new ArrayBuffer(length));
  let channels: Float32Array[] = [];
  let sampleRate = buffer.sampleRate;
  let offset = 0;
  let pos = 0;

  function setUint16(data: number) {
    out.setUint16(pos, data, true);
    pos += 2;
  }

  function setUint32(data: number) {
    out.setUint32(pos, data, true);
    pos += 4;
  }

  // write WAVE header
  setUint32(0x46464952); // "RIFF"
  setUint32(length - 8);
  setUint32(0x45564157); // "WAVE"

  setUint32(0x20746d66); // "fmt " chunk
  setUint32(16); // length = 16
  setUint16(1); // PCM
  setUint16(numOfChan);
  setUint32(sampleRate);
  setUint32(sampleRate * 2 * numOfChan); // byte rate
  setUint16(numOfChan * 2); // block align
  setUint16(16); // bits per sample

  setUint32(0x61746164); // "data" chunk
  setUint32(length - pos - 4);

  for (let i = 0; i < buffer.numberOfChannels; i++) {
    channels.push(buffer.getChannelData(i));
  }

  while (offset < buffer.length) {
    for (let i = 0; i < numOfChan; i++) {
      let sample = Math.max(-1, Math.min(1, channels[i][offset]));
      sample = (0.5 + sample < 0 ? sample * 32768 : sample * 32767) | 0;
      out.setInt16(pos, sample, true);
      pos += 2;
    }
    offset++;
  }

  return new Blob([out], { type: 'audio/wav' });
}
