export interface TranscriptSegment {
  id: string;
  startSec: number;
  endSec: number;
  timestamp: string; // e.g. "00:15" or "01:22"
  speaker: string; // e.g. "Спикер 1" or user edited name
  text: string;
}

export interface SummaryResult {
  overview: string;
  keyPoints: string[];
  actionItems: string[];
  sentiment?: string; // "Позитивный", "Деловой", "Нейтральный", "Напряженный"
  topics?: string[];
}

export interface TranscriptionOptions {
  language: string; // 'auto' | 'ru' | 'en' | 'es' | 'de' | 'fr' | 'kk' | 'zh' | string
  enableDiarization: boolean;
  speakerCount?: number;
  mode: 'full' | 'clean' | 'summary_only' | 'subtitles';
  timestampInterval: 'sentence' | 'paragraph' | '30s';
  customGlossary: string;
  promptNotes?: string;
}

export interface TranscriptRecord {
  id: string;
  title: string;
  fileName: string;
  fileType: 'audio' | 'video';
  fileSize: number;
  durationSec: number;
  mediaUrl?: string; // Blob or Data URL for preview/playback
  mimeType: string;
  createdAt: string;
  languageDetected: string;
  segments: TranscriptSegment[];
  summary: SummaryResult;
  srtContent: string;
  rawText: string;
  wordCount: number;
}

export interface AudioSample {
  id: string;
  title: string;
  category: string;
  duration: string;
  description: string;
  mimeType: string;
  base64Data?: string;
  sampleAudioUrl?: string;
  sampleVideoUrl?: string;
}
