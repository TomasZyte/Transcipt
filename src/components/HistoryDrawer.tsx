import React from 'react';
import { History, FileText, Trash2, ArrowRight, Clock, Calendar, Sparkles } from 'lucide-react';
import { TranscriptRecord } from '../types';

interface HistoryDrawerProps {
  records: TranscriptRecord[];
  onSelectRecord: (record: TranscriptRecord) => void;
  onDeleteRecord: (id: string) => void;
  onClearHistory: () => void;
}

export const HistoryDrawer: React.FC<HistoryDrawerProps> = ({
  records,
  onSelectRecord,
  onDeleteRecord,
  onClearHistory,
}) => {
  return (
    <div className="bg-zinc-900/70 backdrop-blur-xl border border-zinc-800/90 rounded-[28px] p-6 sm:p-8 shadow-2xl space-y-6">
      
      <div className="flex items-center justify-between pb-4 border-b border-zinc-800/80">
        <div className="flex items-center space-x-3.5">
          <div className="w-12 h-12 rounded-2xl bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 flex items-center justify-center">
            <History className="w-6 h-6" />
          </div>
          <div>
            <h2 className="text-xl font-extrabold text-white">История транскрибаций</h2>
            <p className="text-xs text-zinc-400">
              Сохраненные расшифровки автоматически остаются в локальной памяти браузера
            </p>
          </div>
        </div>

        {records.length > 0 && (
          <button
            onClick={onClearHistory}
            className="px-3.5 py-2 text-xs font-semibold text-rose-400 hover:text-rose-300 hover:bg-rose-500/10 rounded-xl transition"
          >
            Очистить историю
          </button>
        )}
      </div>

      {records.length === 0 ? (
        <div className="text-center py-16 space-y-3">
          <div className="w-16 h-16 rounded-2xl bg-zinc-950 border border-zinc-800 text-zinc-600 flex items-center justify-center mx-auto">
            <FileText className="w-8 h-8" />
          </div>
          <p className="text-base font-bold text-zinc-300">История пока пуста</p>
          <p className="text-xs text-zinc-500 max-w-sm mx-auto">
            Загрузите аудио или видеофайл на вкладке "Расшифровка", чтобы сохранить первую аудиозапись.
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {records.map((rec) => (
            <div
              key={rec.id}
              onClick={() => onSelectRecord(rec)}
              className="p-5 bg-zinc-950/70 border border-zinc-800/90 hover:border-indigo-500/60 rounded-2xl transition-all cursor-pointer flex flex-col justify-between group hover:scale-[1.01]"
            >
              <div className="space-y-2.5">
                <div className="flex items-center justify-between">
                  <span className="text-[10px] uppercase font-bold px-2.5 py-0.5 rounded-full bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
                    {rec.languageDetected || 'Русский'}
                  </span>
                  <span className="text-[11px] text-zinc-500 flex items-center gap-1 font-mono">
                    <Calendar className="w-3 h-3 text-zinc-500" />
                    {new Date(rec.createdAt).toLocaleDateString('ru-RU')}
                  </span>
                </div>

                <h4 className="text-sm font-bold text-white group-hover:text-indigo-300 transition truncate">
                  {rec.fileName}
                </h4>

                <p className="text-xs text-zinc-400 line-clamp-2 leading-relaxed">
                  {rec.summary?.overview || rec.rawText}
                </p>
              </div>

              <div className="pt-4 mt-3 border-t border-zinc-800/80 flex items-center justify-between text-xs text-zinc-400">
                <span className="flex items-center gap-1 font-mono text-[11px] font-medium text-zinc-400">
                  <Clock className="w-3.5 h-3.5 text-indigo-400" />
                  {rec.wordCount} слов
                </span>

                <div className="flex items-center space-x-2">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onDeleteRecord(rec.id);
                    }}
                    className="p-1.5 text-zinc-500 hover:text-rose-400 transition"
                    title="Удалить из истории"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>

                  <span className="text-indigo-400 flex items-center gap-1 font-bold group-hover:translate-x-1 transition-transform">
                    Открыть <ArrowRight className="w-3.5 h-3.5" />
                  </span>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

    </div>
  );
};
