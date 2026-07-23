import React, { useState, useRef } from 'react';
import { 
  Upload, Mic, Play, Settings, Sliders, Volume2, FileVideo, FileAudio, 
  Sparkles, CheckCircle2, AlertCircle, Info, Radio, RefreshCw, Zap, ShieldCheck, Cpu, HardDrive
} from 'lucide-react';
import { TranscriptionOptions, AudioSample } from '../types';
import { DEMO_SAMPLES, generateSampleAudioBlob } from '../data/samples';
import { processMediaFileForTranscription } from '../utils/audioExtractor';

interface UploadSectionProps {
  onStartTranscription: (payload: {
    fileData?: string;
    chunks?: any[];
    mimeType: string;
    fileName: string;
    fileType: 'audio' | 'video';
    mediaUrl?: string;
    options: TranscriptionOptions;
    sampleId?: string;
  }) => void;
  isProcessing: boolean;
  processStep: string;
  processPercent: number;
}

export const UploadSection: React.FC<UploadSectionProps> = ({
  onStartTranscription,
  isProcessing,
  processStep,
  processPercent
}) => {
  const [sourceType, setSourceType] = useState<'file' | 'mic' | 'sample'>('file');
  
  // File state
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [filePreviewUrl, setFilePreviewUrl] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Recording state
  const [isRecording, setIsRecording] = useState(false);
  const [recordTimeSec, setRecordTimeSec] = useState(0);
  const [recordedBlob, setRecordedBlob] = useState<Blob | null>(null);
  const [recordedUrl, setRecordedUrl] = useState<string | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<number | null>(null);

  // Sample state
  const [selectedSample, setSelectedSample] = useState<AudioSample>(DEMO_SAMPLES[0]);

  // Options state
  const [options, setOptions] = useState<TranscriptionOptions>({
    language: 'auto',
    enableDiarization: true,
    speakerCount: 2,
    mode: 'full',
    timestampInterval: 'sentence',
    customGlossary: '',
  });

  const [showAdvanced, setShowAdvanced] = useState(false);
  const [isPreparing, setIsPreparing] = useState(false);

  // File drop handlers
  const handleFileChange = (file: File) => {
    setSelectedFile(file);
    const url = URL.createObjectURL(file);
    setFilePreviewUrl(url);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileChange(e.dataTransfer.files[0]);
    }
  };

  // Microphone recording
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        const blob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        setRecordedBlob(blob);
        setRecordedUrl(URL.createObjectURL(blob));
        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorder.start();
      setIsRecording(true);
      setRecordTimeSec(0);

      timerRef.current = window.setInterval(() => {
        setRecordTimeSec(prev => prev + 1);
      }, 1000);
    } catch (err) {
      alert('Не удалось получить доступ к микрофону. Проверьте разрешения браузера.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      if (timerRef.current) clearInterval(timerRef.current);
    }
  };

  const formatTimer = (secs: number) => {
    const m = Math.floor(secs / 60);
    const s = secs % 60;
    return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
  };

  // Process submission
  const handleSubmit = async () => {
    if (isProcessing || isPreparing) return;

    setIsPreparing(true);

    try {
      if (sourceType === 'file') {
        if (!selectedFile) {
          alert('Пожалуйста, выберите аудио или видеофайл');
          setIsPreparing(false);
          return;
        }

        const isVideo = selectedFile.type.startsWith('video/') || selectedFile.name.match(/\.(mp4|webm|mov|avi|mkv)$/i) !== null;
        const processed = await processMediaFileForTranscription(selectedFile);
        
        onStartTranscription({
          fileData: processed.singleBase64,
          chunks: processed.chunks,
          mimeType: processed.singleMimeType || selectedFile.type || (isVideo ? 'video/mp4' : 'audio/mp3'),
          fileName: selectedFile.name,
          fileType: isVideo ? 'video' : 'audio',
          mediaUrl: filePreviewUrl || undefined,
          options,
        });

      } else if (sourceType === 'mic') {
        if (!recordedBlob) {
          alert('Сначала запишите аудио с микрофона');
          setIsPreparing(false);
          return;
        }

        const reader = new FileReader();
        reader.onload = () => {
          const base64 = reader.result as string;
          onStartTranscription({
            fileData: base64,
            mimeType: 'audio/webm',
            fileName: `Запись_микрофон_${new Date().toLocaleTimeString('ru-RU').replace(/:/g, '-')}.webm`,
            fileType: 'audio',
            mediaUrl: recordedUrl || undefined,
            options,
          });
        };
        reader.readAsDataURL(recordedBlob);

      } else if (sourceType === 'sample') {
        const { blob, mimeType } = await generateSampleAudioBlob(selectedSample.id);
        const url = URL.createObjectURL(blob);
        const reader = new FileReader();
        reader.onload = () => {
          const base64 = reader.result as string;
          onStartTranscription({
            fileData: base64,
            mimeType,
            fileName: `${selectedSample.title}.wav`,
            fileType: 'audio',
            mediaUrl: url,
            options,
            sampleId: selectedSample.id,
          });
        };
        reader.readAsDataURL(blob);
      }
    } catch (err: any) {
      console.error('Submit error:', err);
      alert('Ошибка при подготовке файла: ' + (err?.message || 'Неизвестная ошибка'));
    } finally {
      setIsPreparing(false);
    }
  };

  return (
    <div className="space-y-6">
      
      {/* Bento Grid Top Stats & Capabilities Row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 sm:gap-4">
        
        {/* Bento Stat 1: Engine */}
        <div className="bg-zinc-900/60 backdrop-blur-xl border border-zinc-800/80 rounded-2xl p-4 flex items-center space-x-3.5 shadow-lg hover:border-zinc-700/80 transition-all">
          <div className="w-10 h-10 rounded-xl bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 flex items-center justify-center shrink-0">
            <Cpu className="w-5 h-5" />
          </div>
          <div className="min-w-0">
            <p className="text-[11px] font-medium text-zinc-400">AI Нейросеть</p>
            <p className="text-xs sm:text-sm font-bold text-white truncate">Gemini 3.6 Flash</p>
          </div>
        </div>

        {/* Bento Stat 2: Accuracy */}
        <div className="bg-zinc-900/60 backdrop-blur-xl border border-zinc-800/80 rounded-2xl p-4 flex items-center space-x-3.5 shadow-lg hover:border-zinc-700/80 transition-all">
          <div className="w-10 h-10 rounded-xl bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 flex items-center justify-center shrink-0">
            <Zap className="w-5 h-5" />
          </div>
          <div className="min-w-0">
            <p className="text-[11px] font-medium text-zinc-400">Точность распознавания</p>
            <p className="text-xs sm:text-sm font-bold text-white truncate">99.4% • Рус/En/Kz</p>
          </div>
        </div>

        {/* Bento Stat 3: Formats */}
        <div className="bg-zinc-900/60 backdrop-blur-xl border border-zinc-800/80 rounded-2xl p-4 flex items-center space-x-3.5 shadow-lg hover:border-zinc-700/80 transition-all">
          <div className="w-10 h-10 rounded-xl bg-cyan-500/10 border border-cyan-500/20 text-cyan-400 flex items-center justify-center shrink-0">
            <HardDrive className="w-5 h-5" />
          </div>
          <div className="min-w-0">
            <p className="text-[11px] font-medium text-zinc-400">Поддержка файлов</p>
            <p className="text-xs sm:text-sm font-bold text-white truncate">MP4, WEBM, MP3, WAV</p>
          </div>
        </div>

        {/* Bento Stat 4: Diarization */}
        <div className="bg-zinc-900/60 backdrop-blur-xl border border-zinc-800/80 rounded-2xl p-4 flex items-center space-x-3.5 shadow-lg hover:border-zinc-700/80 transition-all">
          <div className="w-10 h-10 rounded-xl bg-amber-500/10 border border-amber-500/20 text-amber-400 flex items-center justify-center shrink-0">
            <ShieldCheck className="w-5 h-5" />
          </div>
          <div className="min-w-0">
            <p className="text-[11px] font-medium text-zinc-400">Конфиденциально</p>
            <p className="text-xs sm:text-sm font-bold text-white truncate">Локальный буфер</p>
          </div>
        </div>

      </div>

      {/* Primary Bento Upload Box */}
      <div className="bg-zinc-900/70 backdrop-blur-xl border border-zinc-800/90 rounded-[28px] sm:rounded-[32px] p-6 sm:p-8 shadow-2xl space-y-6 relative overflow-hidden">
        
        {/* Source Selector Header */}
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 pb-4 border-b border-zinc-800/80">
          <div>
            <h2 className="text-xl font-bold text-white flex items-center gap-2">
              <span>Загрузка медиафайла</span>
              <span className="text-[11px] bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 font-semibold px-2.5 py-0.5 rounded-full">
                Аудио & Видео
              </span>
            </h2>
            <p className="text-xs text-zinc-400 mt-1">
              Выберите медиафайл с устройства, микрофон или готовый демо-фрагмент
            </p>
          </div>

          <div className="flex bg-zinc-950 p-1 rounded-2xl border border-zinc-800/80">
            <button
              onClick={() => setSourceType('file')}
              className={`px-3.5 py-2 rounded-xl text-xs font-semibold transition-all flex items-center gap-1.5 ${
                sourceType === 'file'
                  ? 'bg-zinc-100 text-zinc-950 font-bold shadow'
                  : 'text-zinc-400 hover:text-white'
              }`}
            >
              <Upload className="w-3.5 h-3.5" />
              <span>Файл</span>
            </button>

            <button
              onClick={() => setSourceType('mic')}
              className={`px-3.5 py-2 rounded-xl text-xs font-semibold transition-all flex items-center gap-1.5 ${
                sourceType === 'mic'
                  ? 'bg-zinc-100 text-zinc-950 font-bold shadow'
                  : 'text-zinc-400 hover:text-white'
              }`}
            >
              <Mic className="w-3.5 h-3.5" />
              <span>Микрофон</span>
            </button>

            <button
              onClick={() => setSourceType('sample')}
              className={`px-3.5 py-2 rounded-xl text-xs font-semibold transition-all flex items-center gap-1.5 ${
                sourceType === 'sample'
                  ? 'bg-zinc-100 text-zinc-950 font-bold shadow'
                  : 'text-zinc-400 hover:text-white'
              }`}
            >
              <Sparkles className="w-3.5 h-3.5 text-amber-500" />
              <span>Демо-образцы</span>
            </button>
          </div>
        </div>

        {/* Dropzone for Files */}
        {sourceType === 'file' && (
          <div
            onDragOver={(e) => e.preventDefault()}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className={`border-2 border-dashed rounded-3xl p-8 sm:p-12 text-center transition-all cursor-pointer relative group ${
              selectedFile
                ? 'border-indigo-500/60 bg-indigo-950/20'
                : 'border-zinc-800 hover:border-indigo-500/50 bg-zinc-950/50 hover:bg-zinc-950/80'
            }`}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="audio/*,video/*,.mp3,.wav,.m4a,.ogg,.flac,.aac,.mp4,.webm,.mov,.avi,.mkv"
              className="hidden"
              onChange={(e) => e.target.files?.[0] && handleFileChange(e.target.files[0])}
            />

            {!selectedFile ? (
              <div className="space-y-4">
                <div className="w-16 h-16 rounded-2xl bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 flex items-center justify-center mx-auto group-hover:scale-110 transition-transform shadow-lg">
                  <Upload className="w-8 h-8" />
                </div>
                <div>
                  <p className="text-base font-bold text-zinc-100">
                    Перетащите аудио или видео сюда или <span className="text-indigo-400 underline decoration-indigo-400/40">выберите файл</span>
                  </p>
                  <p className="text-xs text-zinc-400 mt-1">
                    Поддерживаются MP3, WAV, M4A, OGG, MP4, WEBM, MOV, AVI, MKV (до 200 МБ)
                  </p>
                </div>
                <div className="flex flex-wrap items-center justify-center gap-2 pt-2">
                  <span className="text-[11px] px-3 py-1 rounded-lg bg-zinc-800/80 text-zinc-300 font-medium flex items-center gap-1 border border-zinc-700/50">
                    <FileAudio className="w-3.5 h-3.5 text-indigo-400" /> Аудиоклипы
                  </span>
                  <span className="text-[11px] px-3 py-1 rounded-lg bg-zinc-800/80 text-zinc-300 font-medium flex items-center gap-1 border border-zinc-700/50">
                    <FileVideo className="w-3.5 h-3.5 text-emerald-400" /> Видеозаписи
                  </span>
                </div>
              </div>
            ) : (
              <div className="space-y-4" onClick={(e) => e.stopPropagation()}>
                <div className="flex items-center justify-between p-4 bg-zinc-950/80 rounded-2xl border border-zinc-800">
                  <div className="flex items-center space-x-3.5">
                    <div className="w-12 h-12 rounded-xl bg-indigo-500/20 text-indigo-400 flex items-center justify-center shrink-0">
                      {selectedFile.type.startsWith('video/') || selectedFile.name.match(/\.(mp4|webm|mov|avi|mkv)$/i) ? (
                        <FileVideo className="w-6 h-6 text-emerald-400" />
                      ) : (
                        <FileAudio className="w-6 h-6 text-indigo-400" />
                      )}
                    </div>
                    <div className="text-left">
                      <p className="text-sm font-bold text-white truncate max-w-xs sm:max-w-md">
                        {selectedFile.name}
                      </p>
                      <p className="text-xs text-zinc-400">
                        {(selectedFile.size / (1024 * 1024)).toFixed(2)} МБ • {selectedFile.type || 'Медиафайл'}
                      </p>
                    </div>
                  </div>

                  <button
                    onClick={() => {
                      setSelectedFile(null);
                      setFilePreviewUrl(null);
                    }}
                    className="px-3.5 py-1.5 text-xs font-semibold text-rose-400 hover:text-rose-300 hover:bg-rose-500/10 rounded-xl transition"
                  >
                    Удалить
                  </button>
                </div>

                {/* Audio/Video Media Preview */}
                {filePreviewUrl && (
                  <div className="pt-2">
                    {selectedFile.type.startsWith('video/') || selectedFile.name.match(/\.(mp4|webm|mov|avi|mkv)$/i) ? (
                      <video src={filePreviewUrl} controls className="max-h-56 rounded-2xl mx-auto border border-zinc-800 shadow-xl" />
                    ) : (
                      <audio src={filePreviewUrl} controls className="w-full rounded-xl" />
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* Microphone Panel */}
        {sourceType === 'mic' && (
          <div className="bg-zinc-950/80 border border-zinc-800 rounded-3xl p-8 text-center space-y-4">
            <div className="relative inline-block">
              <button
                onClick={isRecording ? stopRecording : startRecording}
                className={`w-20 h-20 rounded-3xl flex items-center justify-center transition-all ${
                  isRecording
                    ? 'bg-rose-600 text-white animate-pulse shadow-xl shadow-rose-600/40'
                    : 'bg-indigo-600 hover:bg-indigo-500 text-white shadow-xl shadow-indigo-600/30'
                }`}
              >
                <Mic className="w-8 h-8" />
              </button>
            </div>

            <div>
              <p className="text-base font-bold text-white">
                {isRecording ? 'Запись микрофона идет...' : recordedBlob ? 'Аудиозапись готова к отправке' : 'Нажмите кнопку для начала записи'}
              </p>
              {isRecording && (
                <p className="text-2xl font-mono font-bold text-rose-400 mt-2">
                  {formatTimer(recordTimeSec)}
                </p>
              )}
            </div>

            {recordedUrl && !isRecording && (
              <div className="pt-2 max-w-md mx-auto">
                <audio src={recordedUrl} controls className="w-full" />
              </div>
            )}
          </div>
        )}

        {/* Sample Select Panel */}
        {sourceType === 'sample' && (
          <div className="space-y-3">
            <p className="text-xs font-semibold text-zinc-400 uppercase tracking-wider">
              Готовые демонстрационные фрагменты для мгновенной проверки:
            </p>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              {DEMO_SAMPLES.map((sample) => (
                <div
                  key={sample.id}
                  onClick={() => setSelectedSample(sample)}
                  className={`p-4 rounded-2xl border transition-all cursor-pointer flex items-start space-x-3.5 ${
                    selectedSample.id === sample.id
                      ? 'border-indigo-500/80 bg-indigo-950/30 shadow-lg'
                      : 'border-zinc-800 bg-zinc-950/40 hover:border-zinc-700'
                  }`}
                >
                  <div className="w-10 h-10 rounded-xl bg-indigo-500/20 text-indigo-400 flex items-center justify-center shrink-0 mt-0.5">
                    <Sparkles className="w-5 h-5 text-amber-400" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between">
                      <span className="text-[11px] text-indigo-400 font-bold">{sample.category}</span>
                      <span className="text-[11px] text-zinc-500 font-mono">{sample.duration}</span>
                    </div>
                    <h4 className="text-sm font-bold text-white truncate mt-0.5">{sample.title}</h4>
                    <p className="text-xs text-zinc-400 line-clamp-2 mt-1">{sample.description}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Options Bento Container */}
        <div className="bg-zinc-950/90 border border-zinc-800/80 rounded-2xl p-4 sm:p-5 space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2 text-zinc-200 font-bold text-xs uppercase tracking-wider">
              <Sliders className="w-4 h-4 text-indigo-400" />
              <span>Параметры транскрибации</span>
            </div>

            <button
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="text-xs text-indigo-400 hover:text-indigo-300 flex items-center gap-1 font-semibold"
            >
              <span>{showAdvanced ? 'Свернуть' : 'Расширенные настройки'}</span>
            </button>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3.5">
            {/* Language */}
            <div>
              <label className="block text-[11px] font-semibold text-zinc-400 mb-1">
                Язык речи
              </label>
              <select
                value={options.language}
                onChange={(e) => setOptions({ ...options, language: e.target.value })}
                className="w-full bg-zinc-900 border border-zinc-800 rounded-xl px-3 py-2 text-xs text-zinc-200 focus:outline-none focus:border-indigo-500"
              >
                <option value="auto">🌐 Автоопределение (Рус / En / Kz)</option>
                <option value="Русский">🇷🇺 Русский</option>
                <option value="Английский">🇬🇧 English</option>
                <option value="Казахский">🇰🇿 Қазақша</option>
                <option value="Испанский">🇪🇸 Español</option>
                <option value="Немецкий">🇩🇪 Deutsch</option>
                <option value="Французский">🇫🇷 Français</option>
                <option value="Китайский">🇨🇳 中文</option>
              </select>
            </div>

            {/* Mode */}
            <div>
              <label className="block text-[11px] font-semibold text-zinc-400 mb-1">
                Режим расшифровки
              </label>
              <select
                value={options.mode}
                onChange={(e) => setOptions({ ...options, mode: e.target.value as any })}
                className="w-full bg-zinc-900 border border-zinc-800 rounded-xl px-3 py-2 text-xs text-zinc-200 focus:outline-none focus:border-indigo-500"
              >
                <option value="full">📝 Дословный текст (Сохраняя всё)</option>
                <option value="clean">✨ Очищенная речь (Без "э-э")</option>
                <option value="summary_only">📊 Саммари и ключевые тезисы</option>
                <option value="subtitles">🎬 Субтитры .SRT</option>
              </select>
            </div>

            {/* Diarization */}
            <div>
              <label className="block text-[11px] font-semibold text-zinc-400 mb-1">
                Разделение спикеров
              </label>
              <button
                onClick={() => setOptions({ ...options, enableDiarization: !options.enableDiarization })}
                className={`w-full py-2 px-3 rounded-xl text-xs font-semibold border flex items-center justify-between transition-colors ${
                  options.enableDiarization
                    ? 'bg-indigo-950/60 border-indigo-500/50 text-indigo-300'
                    : 'bg-zinc-900 border-zinc-800 text-zinc-400'
                }`}
              >
                <span>{options.enableDiarization ? '👥 Включено (Спикер 1, 2...)' : '👤 Монолог'}</span>
                <CheckCircle2 className={`w-4 h-4 ${options.enableDiarization ? 'text-indigo-400' : 'text-zinc-600'}`} />
              </button>
            </div>
          </div>

          {/* Custom Glossary Field */}
          {showAdvanced && (
            <div className="pt-2 border-t border-zinc-800/80 space-y-2">
              <label className="block text-[11px] font-semibold text-zinc-400">
                Словарь терминов (имена, бренды, профессиональный сленг)
              </label>
              <input
                type="text"
                placeholder="Например: Илон Маск, Gemini API, Kubernetes, FinTech, Реактор"
                value={options.customGlossary}
                onChange={(e) => setOptions({ ...options, customGlossary: e.target.value })}
                className="w-full bg-zinc-900 border border-zinc-800 rounded-xl px-3 py-2 text-xs text-zinc-200 focus:outline-none focus:border-indigo-500"
              />
            </div>
          )}
        </div>

        {/* Progress Bar Display while processing */}
        {isProcessing && (
          <div className="p-5 bg-indigo-950/60 border border-indigo-500/40 rounded-2xl space-y-3 shadow-xl animate-pulse">
            <div className="flex items-center justify-between text-xs font-bold text-indigo-300">
              <span className="flex items-center gap-2">
                <RefreshCw className="w-4 h-4 animate-spin text-indigo-400" />
                {processStep || 'Расшифровка медиафайла нейросетью...'}
              </span>
              <span>{processPercent}%</span>
            </div>
            <div className="w-full h-2.5 bg-zinc-900 rounded-full overflow-hidden border border-zinc-800">
              <div
                className="h-full bg-gradient-to-r from-indigo-500 via-indigo-400 to-emerald-400 transition-all duration-300"
                style={{ width: `${processPercent}%` }}
              />
            </div>
          </div>
        )}

        {/* Action Button */}
        <div className="pt-2">
          <button
            onClick={handleSubmit}
            disabled={isProcessing || isPreparing || (sourceType === 'file' && !selectedFile)}
            className={`w-full py-4 px-6 rounded-2xl font-bold text-sm sm:text-base transition-all flex items-center justify-center space-x-2.5 shadow-xl ${
              isProcessing || isPreparing || (sourceType === 'file' && !selectedFile)
                ? 'bg-zinc-800 text-zinc-500 cursor-not-allowed border border-zinc-700/50'
                : 'bg-zinc-100 hover:bg-white text-zinc-950 hover:scale-[1.01] active:scale-[0.99] shadow-indigo-500/20'
            }`}
          >
            {isProcessing || isPreparing ? (
              <>
                <RefreshCw className="w-5 h-5 animate-spin text-indigo-600" />
                <span>{isPreparing ? 'Подготовка медиафайла...' : 'Обработка Gemini AI...'}</span>
              </>
            ) : (
              <>
                <Sparkles className="w-5 h-5 text-indigo-600" />
                <span>Начать расшифровку {sourceType === 'sample' ? `("${selectedSample.title}")` : ''}</span>
              </>
            )}
          </button>
        </div>

      </div>

    </div>
  );
};
