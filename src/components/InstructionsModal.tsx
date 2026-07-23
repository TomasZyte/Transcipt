import React from 'react';
import { HelpCircle, Mic, Sparkles, FileText, Globe, ListChecks, CheckCircle2, Cpu, Zap, Download } from 'lucide-react';

export const InstructionsModal: React.FC = () => {
  return (
    <div className="bg-zinc-900/70 backdrop-blur-xl border border-zinc-800/90 rounded-[28px] sm:rounded-[32px] p-6 sm:p-8 shadow-2xl space-y-8 max-w-4xl mx-auto">
      
      {/* Header */}
      <div className="text-center space-y-2">
        <div className="w-14 h-14 rounded-2xl bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 flex items-center justify-center mx-auto mb-3 shadow-lg">
          <HelpCircle className="w-7 h-7" />
        </div>
        <h2 className="text-2xl sm:text-3xl font-extrabold text-white">Как работать с Tranzip AI</h2>
        <p className="text-xs sm:text-sm text-zinc-400 max-w-lg mx-auto">
          Автоматическая транскрибация аудио и видео, спикер-диаризация и AI-анализ
        </p>
      </div>

      {/* Steps Bento Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        
        <div className="p-6 bg-zinc-950/80 border border-zinc-800/90 rounded-2xl space-y-3">
          <div className="w-9 h-9 rounded-xl bg-zinc-100 text-zinc-950 font-extrabold flex items-center justify-center text-sm shadow">
            1
          </div>
          <h3 className="text-base font-bold text-white">Загрузка медиа</h3>
          <p className="text-xs text-zinc-400 leading-relaxed">
            Загрузите любой аудио или видеофайл (MP3, WAV, MP4, WEBM, MOV) или записанный с микрофона разговор.
          </p>
        </div>

        <div className="p-6 bg-zinc-950/80 border border-zinc-800/90 rounded-2xl space-y-3">
          <div className="w-9 h-9 rounded-xl bg-zinc-100 text-zinc-950 font-extrabold flex items-center justify-center text-sm shadow">
            2
          </div>
          <h3 className="text-base font-bold text-white">Нейросеть Gemini 3.6</h3>
          <p className="text-xs text-zinc-400 leading-relaxed">
            Модель прослушивает медиапоток, автоматически разделяет спикеров, проставляет таймкоды и формирует резюме.
          </p>
        </div>

        <div className="p-6 bg-zinc-950/80 border border-zinc-800/90 rounded-2xl space-y-3">
          <div className="w-9 h-9 rounded-xl bg-zinc-100 text-zinc-950 font-extrabold flex items-center justify-center text-sm shadow">
            3
          </div>
          <h3 className="text-base font-bold text-white">Редактирование & Экспорт</h3>
          <p className="text-xs text-zinc-400 leading-relaxed">
            Синхронизируйте плеер с текстом, переименовывайте спикеров и скачивайте субтитры .SRT, .TXT или .MD.
          </p>
        </div>

      </div>

      {/* Pro Tips Bento Box */}
      <div className="bg-zinc-950/90 border border-zinc-800/90 rounded-2xl p-6 space-y-4">
        <h3 className="text-xs font-bold uppercase text-indigo-400 tracking-wider flex items-center gap-2">
          <Sparkles className="w-4 h-4 text-amber-400" /> Полезные возможности
        </h3>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs text-zinc-300">
          <div className="flex items-start space-x-3 bg-zinc-900/60 p-3 rounded-xl border border-zinc-800/80">
            <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
            <span><strong>Глоссарий терминов:</strong> Добавьте специфические названия и имена в настройки перед стартом.</span>
          </div>

          <div className="flex items-start space-x-3 bg-zinc-900/60 p-3 rounded-xl border border-zinc-800/80">
            <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
            <span><strong>Переименование спикеров:</strong> Нажмите на имя спикера в транскрипте, чтобы обновить его во всем файле.</span>
          </div>

          <div className="flex items-start space-x-3 bg-zinc-900/60 p-3 rounded-xl border border-zinc-800/80">
            <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
            <span><strong>Синхронизированный плеер:</strong> Кликните по таймкоду [00:15], чтобы переместить воспроизведение на эту секунду.</span>
          </div>

          <div className="flex items-start space-x-3 bg-zinc-900/60 p-3 rounded-xl border border-zinc-800/80">
            <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
            <span><strong>Субтитры SRT:</strong> Готовый файл субтитров можно импортировать в любой видеоредактор или на YouTube.</span>
          </div>
        </div>
      </div>

    </div>
  );
};
