import React, { useState, useRef } from 'react';
import {
  Play, Pause, RotateCcw, RotateCw, Volume2, Search, Filter, Download, 
  Copy, Edit3, Check, Sparkles, MessageSquare, Globe, FileText, UserCheck, 
  ListChecks, CornerDownRight, RefreshCw, Zap, Clock, ShieldAlert
} from 'lucide-react';
import { TranscriptRecord, TranscriptSegment } from '../types';

interface TranscriptViewProps {
  transcript: TranscriptRecord;
  onUpdateTranscript: (updated: TranscriptRecord) => void;
  onNewTranscription: () => void;
}

export const TranscriptView: React.FC<TranscriptViewProps> = ({
  transcript,
  onUpdateTranscript,
  onNewTranscription,
}) => {
  // Player state
  const mediaRef = useRef<HTMLMediaElement | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(transcript.durationSec || 0);
  const [playbackRate, setPlaybackRate] = useState(1);

  // Search & Filter state
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedSpeakerFilter, setSelectedSpeakerFilter] = useState<string>('all');

  // Editing state
  const [editingSpeakerId, setEditingSpeakerId] = useState<string | null>(null);
  const [speakerNameInput, setSpeakerNameInput] = useState('');
  const [copySuccess, setCopySuccess] = useState(false);

  // AI Assistant side panel tab
  const [activeSideTab, setActiveSideTab] = useState<'summary' | 'ai_chat' | 'translate' | 'export'>('summary');
  const [aiQuestion, setAiQuestion] = useState('');
  const [aiAnswer, setAiAnswer] = useState<string | null>(null);
  const [isAiLoading, setIsAiLoading] = useState(false);

  // Translation state
  const [targetLang, setTargetLang] = useState('English');
  const [translatedText, setTranslatedText] = useState<string | null>(null);
  const [isTranslating, setIsTranslating] = useState(false);

  // Sync player time
  const handleTimeUpdate = () => {
    if (mediaRef.current) {
      setCurrentTime(mediaRef.current.currentTime);
    }
  };

  const handleLoadedMetadata = () => {
    if (mediaRef.current && mediaRef.current.duration) {
      setDuration(mediaRef.current.duration);
    }
  };

  const togglePlay = () => {
    if (mediaRef.current) {
      if (isPlaying) {
        mediaRef.current.pause();
      } else {
        mediaRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  const seekToSec = (sec: number) => {
    if (mediaRef.current) {
      mediaRef.current.currentTime = sec;
      setCurrentTime(sec);
      if (!isPlaying) {
        mediaRef.current.play();
        setIsPlaying(true);
      }
    }
  };

  const changeRate = (rate: number) => {
    setPlaybackRate(rate);
    if (mediaRef.current) {
      mediaRef.current.playbackRate = rate;
    }
  };

  const skipSeconds = (secs: number) => {
    if (mediaRef.current) {
      mediaRef.current.currentTime = Math.max(0, Math.min(duration, mediaRef.current.currentTime + secs));
    }
  };

  // Speakers list
  const uniqueSpeakers = Array.from(new Set((transcript.segments || []).map((s) => s.speaker)));

  // Filter segments
  const filteredSegments = (transcript.segments || []).filter((seg) => {
    const matchesSpeaker = selectedSpeakerFilter === 'all' || seg.speaker === selectedSpeakerFilter;
    const matchesQuery = !searchQuery || seg.text.toLowerCase().includes(searchQuery.toLowerCase()) || seg.speaker.toLowerCase().includes(searchQuery.toLowerCase());
    return matchesSpeaker && matchesQuery;
  });

  // Rename speaker across all segments
  const handleRenameSpeaker = (oldName: string) => {
    if (!speakerNameInput.trim() || speakerNameInput === oldName) {
      setEditingSpeakerId(null);
      return;
    }
    const updatedSegments = transcript.segments.map((seg) => {
      if (seg.speaker === oldName) {
        return { ...seg, speaker: speakerNameInput.trim() };
      }
      return seg;
    });
    onUpdateTranscript({ ...transcript, segments: updatedSegments });
    setEditingSpeakerId(null);
  };

  // Edit segment text
  const handleUpdateSegmentText = (segId: string, newText: string) => {
    const updatedSegments = transcript.segments.map((seg) => {
      if (seg.id === segId) {
        return { ...seg, text: newText };
      }
      return seg;
    });
    const rawText = updatedSegments.map((s) => `${s.speaker}: ${s.text}`).join('\n\n');
    onUpdateTranscript({ ...transcript, segments: updatedSegments, rawText });
  };

  // Copy full transcript
  const copyFullTranscript = () => {
    const full = (transcript.segments || []).map((s) => `[${s.timestamp}] ${s.speaker}: ${s.text}`).join('\n\n');
    navigator.clipboard.writeText(full);
    setCopySuccess(true);
    setTimeout(() => setCopySuccess(false), 2000);
  };

  // Download SRT
  const downloadFile = (content: string, filename: string, mime: string) => {
    const blob = new Blob([content], { type: mime });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  };

  // AI Query call
  const handleAskAi = async (promptText?: string) => {
    const query = promptText || aiQuestion;
    if (!query) return;

    setIsAiLoading(true);
    try {
      const res = await fetch('/api/analyze-transcript', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          transcriptText: transcript.rawText,
          segments: transcript.segments,
          action: promptText ? (promptText.includes('протокол') ? 'protocol' : 'question') : 'question',
          customPrompt: query,
        }),
      });
      const data = await res.json();
      if (data.success) {
        setAiAnswer(data.result);
      }
    } catch (e) {
      setAiAnswer('Произошла ошибка при обращении к AI.');
    } finally {
      setIsAiLoading(false);
    }
  };

  // Translation call
  const handleTranslate = async () => {
    setIsTranslating(true);
    try {
      const res = await fetch('/api/analyze-transcript', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          transcriptText: transcript.rawText,
          action: 'translate',
          targetLanguage: targetLang,
        }),
      });
      const data = await res.json();
      if (data.success) {
        setTranslatedText(data.result);
      }
    } catch (e) {
      alert('Ошибка при переводе транскрипта');
    } finally {
      setIsTranslating(false);
    }
  };

  const formatSec = (secs: number) => {
    const m = Math.floor(secs / 60);
    const s = Math.floor(secs % 60);
    return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
  };

  return (
    <div className="space-y-6">
      
      {/* Top Bento Header Bar */}
      <div className="bg-zinc-900/70 backdrop-blur-xl border border-zinc-800/90 rounded-[28px] p-5 sm:p-6 shadow-2xl flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <div className="flex items-center space-x-2">
            <span className="text-[11px] font-bold uppercase px-2.5 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
              {transcript.languageDetected || 'Русский'}
            </span>
            <span className="text-xs text-zinc-400 flex items-center gap-1 font-mono">
              <Clock className="w-3.5 h-3.5 text-indigo-400" /> {transcript.wordCount} слов • ~{formatSec(transcript.durationSec)}
            </span>
          </div>
          <h2 className="text-xl sm:text-2xl font-extrabold text-white mt-1.5 truncate max-w-xl">
            {transcript.fileName || 'Расшифрованный транскрипт'}
          </h2>
        </div>

        <div className="flex items-center space-x-2">
          <button
            onClick={copyFullTranscript}
            className="px-4 py-2.5 rounded-2xl bg-zinc-800 hover:bg-zinc-700 text-zinc-200 text-xs font-bold flex items-center gap-1.5 transition border border-zinc-700/60"
          >
            {copySuccess ? <Check className="w-4 h-4 text-emerald-400" /> : <Copy className="w-4 h-4" />}
            <span>{copySuccess ? 'Скопировано!' : 'Копировать'}</span>
          </button>

          <button
            onClick={onNewTranscription}
            className="px-4 py-2.5 rounded-2xl bg-zinc-100 hover:bg-white text-zinc-950 text-xs font-bold flex items-center gap-1.5 transition shadow-lg hover:scale-[1.02]"
          >
            <Sparkles className="w-4 h-4 text-indigo-600" />
            <span>Новая запись</span>
          </button>
        </div>
      </div>

      {/* Player Bento Card */}
      <div className="bg-zinc-900/70 backdrop-blur-xl border border-zinc-800/90 rounded-[28px] p-5 sm:p-6 shadow-2xl space-y-4">
        
        {/* HTML Media Player */}
        {transcript.mediaUrl && (
          <div>
            {transcript.fileType === 'video' ? (
              <video
                ref={mediaRef as any}
                src={transcript.mediaUrl}
                onTimeUpdate={handleTimeUpdate}
                onLoadedMetadata={handleLoadedMetadata}
                onEnded={() => setIsPlaying(false)}
                className="max-h-72 rounded-2xl mx-auto border border-zinc-800 shadow-2xl"
              />
            ) : (
              <audio
                ref={mediaRef as any}
                src={transcript.mediaUrl}
                onTimeUpdate={handleTimeUpdate}
                onLoadedMetadata={handleLoadedMetadata}
                onEnded={() => setIsPlaying(false)}
                className="hidden"
              />
            )}
          </div>
        )}

        {/* Equalizer Visualizer for Audio */}
        {transcript.fileType !== 'video' && (
          <div className="bg-zinc-950 p-3.5 rounded-2xl border border-zinc-800/80 flex items-center justify-center space-x-1 h-14 overflow-hidden">
            {Array.from({ length: 42 }).map((_, i) => {
              const active = isPlaying;
              const height = active ? Math.max(15, (Math.sin(i * 0.4 + currentTime * 5) * 0.5 + 0.5) * 100) : 25;
              return (
                <div
                  key={i}
                  className={`w-1 rounded-full transition-all duration-150 ${
                    active ? 'bg-indigo-500' : 'bg-zinc-800'
                  }`}
                  style={{ height: `${height}%` }}
                />
              );
            })}
          </div>
        )}

        {/* Player Controls Bar */}
        <div className="flex flex-col sm:flex-row items-center justify-between gap-4 pt-1">
          <div className="flex items-center space-x-3">
            <button
              onClick={() => skipSeconds(-10)}
              className="p-2.5 text-zinc-400 hover:text-white bg-zinc-800 hover:bg-zinc-700/80 rounded-xl transition"
              title="-10 секунд"
            >
              <RotateCcw className="w-4 h-4" />
            </button>

            <button
              onClick={togglePlay}
              className="w-12 h-12 rounded-2xl bg-zinc-100 hover:bg-white text-zinc-950 flex items-center justify-center shadow-lg shadow-white/10 transition hover:scale-105"
            >
              {isPlaying ? <Pause className="w-6 h-6" /> : <Play className="w-6 h-6 ml-0.5" />}
            </button>

            <button
              onClick={() => skipSeconds(10)}
              className="p-2.5 text-zinc-400 hover:text-white bg-zinc-800 hover:bg-zinc-700/80 rounded-xl transition"
              title="+10 секунд"
            >
              <RotateCw className="w-4 h-4" />
            </button>

            <span className="text-xs font-mono font-bold text-zinc-300 min-w-[90px]">
              {formatSec(currentTime)} / {formatSec(duration)}
            </span>
          </div>

          {/* Time Slider */}
          <div className="flex-1 w-full mx-2">
            <input
              type="range"
              min={0}
              max={duration || 100}
              value={currentTime}
              onChange={(e) => seekToSec(Number(e.target.value))}
              className="w-full h-2 bg-zinc-800 rounded-lg appearance-none cursor-pointer accent-indigo-500"
            />
          </div>

          {/* Speed Buttons */}
          <div className="flex items-center space-x-1 bg-zinc-950 p-1 rounded-2xl border border-zinc-800">
            {[0.75, 1, 1.25, 1.5, 2].map((rate) => (
              <button
                key={rate}
                onClick={() => changeRate(rate)}
                className={`px-2.5 py-1 rounded-xl text-xs font-bold font-mono transition ${
                  playbackRate === rate
                    ? 'bg-zinc-100 text-zinc-950'
                    : 'text-zinc-400 hover:text-white'
                }`}
              >
                {rate}x
              </button>
            ))}
          </div>
        </div>

      </div>

      {/* Main Grid: Left side transcript editor, Right side AI tools */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* Left Column: Transcript Editor (7 cols) */}
        <div className="lg:col-span-7 bg-zinc-900/70 backdrop-blur-xl border border-zinc-800/90 rounded-[28px] p-5 space-y-4 shadow-2xl">
          
          {/* Search & Speaker Filter */}
          <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3 pb-3 border-b border-zinc-800/80">
            <div className="relative flex-1">
              <Search className="w-4 h-4 text-zinc-400 absolute left-3.5 top-2.5" />
              <input
                type="text"
                placeholder="Поиск по тексту или спикеру..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full bg-zinc-950 border border-zinc-800 rounded-xl pl-9 pr-3 py-2 text-xs text-white placeholder-zinc-500 focus:outline-none focus:border-indigo-500"
              />
            </div>

            <div className="flex items-center space-x-2">
              <Filter className="w-4 h-4 text-zinc-400" />
              <select
                value={selectedSpeakerFilter}
                onChange={(e) => setSelectedSpeakerFilter(e.target.value)}
                className="bg-zinc-950 border border-zinc-800 rounded-xl px-3 py-2 text-xs text-zinc-300 focus:outline-none focus:border-indigo-500"
              >
                <option value="all">Все спикеры</option>
                {uniqueSpeakers.map((sp) => (
                  <option key={sp} value={sp}>
                    {sp}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Segments List */}
          <div className="space-y-3 max-h-[600px] overflow-y-auto pr-1">
            {filteredSegments.length === 0 ? (
              <p className="text-xs text-zinc-500 text-center py-12">
                Ничего не найдено по вашему запросу.
              </p>
            ) : (
              filteredSegments.map((seg) => {
                const isActive = currentTime >= seg.startSec && currentTime <= seg.endSec;

                return (
                  <div
                    key={seg.id}
                    className={`p-4 rounded-2xl border transition-all ${
                      isActive
                        ? 'bg-indigo-950/40 border-indigo-500/80 shadow-lg'
                        : 'bg-zinc-950/60 border-zinc-800/80 hover:border-zinc-700'
                    }`}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center space-x-2">
                        {/* Timestamp Button */}
                        <button
                          onClick={() => seekToSec(seg.startSec)}
                          className={`text-xs font-mono font-bold px-2.5 py-1 rounded-lg transition ${
                            isActive
                              ? 'bg-indigo-600 text-white'
                              : 'bg-zinc-800/80 text-indigo-400 hover:bg-indigo-600/30'
                          }`}
                          title="Перейти к этому моменту"
                        >
                          {seg.timestamp}
                        </button>

                        {/* Speaker Name Tag */}
                        {editingSpeakerId === seg.speaker ? (
                          <div className="flex items-center space-x-1">
                            <input
                              type="text"
                              value={speakerNameInput}
                              onChange={(e) => setSpeakerNameInput(e.target.value)}
                              className="bg-zinc-900 border border-indigo-500 text-xs px-2 py-0.5 rounded-lg text-white focus:outline-none"
                              autoFocus
                            />
                            <button
                              onClick={() => handleRenameSpeaker(seg.speaker)}
                              className="p-1 bg-emerald-600 text-white rounded-lg hover:bg-emerald-500"
                            >
                              <Check className="w-3.5 h-3.5" />
                            </button>
                          </div>
                        ) : (
                          <button
                            onClick={() => {
                              setEditingSpeakerId(seg.speaker);
                              setSpeakerNameInput(seg.speaker);
                            }}
                            className="text-xs font-bold text-zinc-300 hover:text-indigo-300 flex items-center gap-1 group"
                            title="Переименовать спикера везде"
                          >
                            <UserCheck className="w-3.5 h-3.5 text-indigo-400" />
                            <span>{seg.speaker}</span>
                            <Edit3 className="w-3 h-3 opacity-0 group-hover:opacity-100 text-zinc-500" />
                          </button>
                        )}
                      </div>

                      <button
                        onClick={() => navigator.clipboard.writeText(seg.text)}
                        className="text-[11px] text-zinc-500 hover:text-zinc-300 transition"
                        title="Скопировать абзац"
                      >
                        <Copy className="w-3.5 h-3.5" />
                      </button>
                    </div>

                    {/* Segment Editable Text */}
                    <textarea
                      value={seg.text}
                      onChange={(e) => handleUpdateSegmentText(seg.id, e.target.value)}
                      rows={2}
                      className="w-full bg-transparent text-xs text-zinc-200 focus:outline-none focus:bg-zinc-900/80 p-2 rounded-xl border border-transparent focus:border-zinc-700 transition resize-y leading-relaxed"
                    />
                  </div>
                );
              })
            )}
          </div>

        </div>

        {/* Right Column: AI Analytics & Export Tools (5 cols) */}
        <div className="lg:col-span-5 bg-zinc-900/70 backdrop-blur-xl border border-zinc-800/90 rounded-[28px] p-5 space-y-4 shadow-2xl">
          
          {/* Side Tabs */}
          <div className="flex bg-zinc-950 p-1 rounded-2xl border border-zinc-800">
            <button
              onClick={() => setActiveSideTab('summary')}
              className={`flex-1 py-1.5 rounded-xl text-xs font-bold transition flex items-center justify-center gap-1 ${
                activeSideTab === 'summary'
                  ? 'bg-zinc-100 text-zinc-950'
                  : 'text-zinc-400 hover:text-white'
              }`}
            >
              <Sparkles className="w-3.5 h-3.5 text-amber-500" />
              <span>Саммари</span>
            </button>

            <button
              onClick={() => setActiveSideTab('ai_chat')}
              className={`flex-1 py-1.5 rounded-xl text-xs font-bold transition flex items-center justify-center gap-1 ${
                activeSideTab === 'ai_chat'
                  ? 'bg-zinc-100 text-zinc-950'
                  : 'text-zinc-400 hover:text-white'
              }`}
            >
              <MessageSquare className="w-3.5 h-3.5 text-indigo-500" />
              <span>AI Чат</span>
            </button>

            <button
              onClick={() => setActiveSideTab('translate')}
              className={`flex-1 py-1.5 rounded-xl text-xs font-bold transition flex items-center justify-center gap-1 ${
                activeSideTab === 'translate'
                  ? 'bg-zinc-100 text-zinc-950'
                  : 'text-zinc-400 hover:text-white'
              }`}
            >
              <Globe className="w-3.5 h-3.5 text-cyan-500" />
              <span>Перевод</span>
            </button>

            <button
              onClick={() => setActiveSideTab('export')}
              className={`flex-1 py-1.5 rounded-xl text-xs font-bold transition flex items-center justify-center gap-1 ${
                activeSideTab === 'export'
                  ? 'bg-zinc-100 text-zinc-950'
                  : 'text-zinc-400 hover:text-white'
              }`}
            >
              <Download className="w-3.5 h-3.5 text-emerald-500" />
              <span>Экспорт</span>
            </button>
          </div>

          {/* TAB 1: Summary */}
          {activeSideTab === 'summary' && (
            <div className="space-y-4">
              <div className="p-4 bg-zinc-950 border border-zinc-800 rounded-2xl space-y-2">
                <h4 className="text-xs font-bold uppercase text-indigo-400 tracking-wider flex items-center gap-1.5">
                  <FileText className="w-4 h-4" /> Краткое содержание
                </h4>
                <p className="text-xs text-zinc-300 leading-relaxed">
                  {transcript.summary?.overview || 'Анализ содержания сформирован.'}
                </p>
                {transcript.summary?.sentiment && (
                  <div className="pt-2 flex items-center space-x-2 text-[11px] text-zinc-400">
                    <span>Эмоциональный тон:</span>
                    <span className="font-semibold text-emerald-400 px-2 py-0.5 rounded-lg bg-emerald-500/10 border border-emerald-500/20">
                      {transcript.summary.sentiment}
                    </span>
                  </div>
                )}
              </div>

              {transcript.summary?.keyPoints && transcript.summary.keyPoints.length > 0 && (
                <div className="p-4 bg-zinc-950 border border-zinc-800 rounded-2xl space-y-2">
                  <h4 className="text-xs font-bold uppercase text-amber-400 tracking-wider flex items-center gap-1.5">
                    <Sparkles className="w-4 h-4" /> Главные тезисы
                  </h4>
                  <ul className="space-y-1.5">
                    {transcript.summary.keyPoints.map((point, i) => (
                      <li key={i} className="text-xs text-zinc-300 flex items-start space-x-2">
                        <span className="text-amber-400 mt-0.5">•</span>
                        <span>{point}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {transcript.summary?.actionItems && transcript.summary.actionItems.length > 0 && (
                <div className="p-4 bg-zinc-950 border border-zinc-800 rounded-2xl space-y-2">
                  <h4 className="text-xs font-bold uppercase text-emerald-400 tracking-wider flex items-center gap-1.5">
                    <ListChecks className="w-4 h-4" /> Поручения и задачи
                  </h4>
                  <ul className="space-y-2">
                    {transcript.summary.actionItems.map((item, i) => (
                      <li key={i} className="text-xs text-zinc-300 flex items-start space-x-2 bg-zinc-900/80 p-2.5 rounded-xl border border-zinc-800">
                        <input type="checkbox" className="mt-0.5 accent-emerald-500 rounded" />
                        <span>{item}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}

          {/* TAB 2: AI Chat Q&A */}
          {activeSideTab === 'ai_chat' && (
            <div className="space-y-4">
              <p className="text-xs text-zinc-400">
                Задайте любой вопрос или выберите готовый сценарий:
              </p>

              <div className="grid grid-cols-1 gap-2">
                <button
                  onClick={() => handleAskAi('Составь официальный протокол встречи с выводами')}
                  className="p-3 bg-zinc-950 hover:bg-zinc-800 border border-zinc-800 rounded-xl text-xs text-left text-indigo-300 transition flex items-center justify-between"
                >
                  <span>📋 Сформировать протокол встречи</span>
                  <CornerDownRight className="w-3.5 h-3.5" />
                </button>

                <button
                  onClick={() => handleAskAi('Найди в записи все упоминания сроков, дат и денег')}
                  className="p-3 bg-zinc-950 hover:bg-zinc-800 border border-zinc-800 rounded-xl text-xs text-left text-amber-300 transition flex items-center justify-between"
                >
                  <span>💰 Найти цифры, даты и бюджет</span>
                  <CornerDownRight className="w-3.5 h-3.5" />
                </button>
              </div>

              <div className="flex items-center space-x-2">
                <input
                  type="text"
                  placeholder="Ваш вопрос по записи..."
                  value={aiQuestion}
                  onChange={(e) => setAiQuestion(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleAskAi()}
                  className="flex-1 bg-zinc-950 border border-zinc-800 rounded-xl px-3 py-2 text-xs text-white focus:outline-none focus:border-indigo-500"
                />
                <button
                  onClick={() => handleAskAi()}
                  disabled={isAiLoading || !aiQuestion}
                  className="px-3.5 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-xl text-xs font-bold disabled:opacity-50"
                >
                  {isAiLoading ? <RefreshCw className="w-4 h-4 animate-spin" /> : 'Спросить'}
                </button>
              </div>

              {aiAnswer && (
                <div className="p-4 bg-zinc-950 border border-indigo-500/30 rounded-2xl space-y-2">
                  <span className="text-[11px] font-bold text-indigo-400 uppercase">Ответ AI:</span>
                  <p className="text-xs text-zinc-200 whitespace-pre-wrap leading-relaxed">
                    {aiAnswer}
                  </p>
                </div>
              )}
            </div>
          )}

          {/* TAB 3: Translate */}
          {activeSideTab === 'translate' && (
            <div className="space-y-4">
              <div className="p-4 bg-zinc-950 border border-zinc-800 rounded-2xl space-y-3">
                <label className="block text-xs font-bold text-zinc-300">
                  Выберите язык для перевода:
                </label>
                <select
                  value={targetLang}
                  onChange={(e) => setTargetLang(e.target.value)}
                  className="w-full bg-zinc-900 border border-zinc-800 rounded-xl px-3 py-2 text-xs text-zinc-200 focus:outline-none"
                >
                  <option value="English">🇬🇧 English</option>
                  <option value="Русский">🇷🇺 Русский</option>
                  <option value="Казахский">🇰🇿 Қазақша</option>
                  <option value="Испанский">🇪🇸 Español</option>
                  <option value="Немецкий">🇩🇪 Deutsch</option>
                  <option value="Французский">🇫🇷 Français</option>
                  <option value="Китайский">🇨🇳 中文</option>
                </select>

                <button
                  onClick={handleTranslate}
                  disabled={isTranslating}
                  className="w-full py-3 bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-bold rounded-xl transition flex items-center justify-center gap-2"
                >
                  {isTranslating ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Globe className="w-4 h-4" />}
                  <span>Перевести весь транскрипт</span>
                </button>
              </div>

              {translatedText && (
                <div className="p-4 bg-zinc-950 border border-zinc-800 rounded-2xl space-y-2 max-h-[350px] overflow-y-auto">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-bold text-indigo-400 uppercase">
                      Перевод на {targetLang}:
                    </span>
                    <button
                      onClick={() => navigator.clipboard.writeText(translatedText)}
                      className="text-[11px] text-zinc-400 hover:text-white"
                    >
                      Скопировать
                    </button>
                  </div>
                  <p className="text-xs text-zinc-200 whitespace-pre-wrap leading-relaxed">
                    {translatedText}
                  </p>
                </div>
              )}
            </div>
          )}

          {/* TAB 4: Export */}
          {activeSideTab === 'export' && (
            <div className="space-y-3">
              <p className="text-xs text-zinc-400">
                Экспортируйте результат в готовых файловых форматах:
              </p>

              <button
                onClick={() => downloadFile(transcript.srtContent, `${transcript.fileName || 'subtitles'}.srt`, 'text/plain')}
                className="w-full p-3 bg-zinc-950 hover:bg-zinc-800/80 border border-zinc-800 rounded-2xl text-xs text-left transition flex items-center justify-between group"
              >
                <div className="flex items-center space-x-3">
                  <div className="w-9 h-9 rounded-xl bg-emerald-500/20 text-emerald-400 flex items-center justify-center text-lg">
                    🎬
                  </div>
                  <div>
                    <p className="font-bold text-white">Субтитры (.SRT)</p>
                    <p className="text-[11px] text-zinc-400">С таймкодами для видеоредакторов</p>
                  </div>
                </div>
                <Download className="w-4 h-4 text-zinc-400 group-hover:text-white" />
              </button>

              <button
                onClick={() => downloadFile(transcript.rawText, `${transcript.fileName || 'transcript'}.txt`, 'text/plain')}
                className="w-full p-3 bg-zinc-950 hover:bg-zinc-800/80 border border-zinc-800 rounded-2xl text-xs text-left transition flex items-center justify-between group"
              >
                <div className="flex items-center space-x-3">
                  <div className="w-9 h-9 rounded-xl bg-indigo-500/20 text-indigo-400 flex items-center justify-center text-lg">
                    📄
                  </div>
                  <div>
                    <p className="font-bold text-white">Текстовый документ (.TXT)</p>
                    <p className="text-[11px] text-zinc-400">Чистый текст расшифровки</p>
                  </div>
                </div>
                <Download className="w-4 h-4 text-zinc-400 group-hover:text-white" />
              </button>

              <button
                onClick={() => {
                  const md = `# ${transcript.fileName}\n\n**Язык**: ${transcript.languageDetected}\n\n## Транскрипт\n\n` + 
                    transcript.segments.map(s => `### ${s.speaker} [${s.timestamp}]\n${s.text}`).join('\n\n');
                  downloadFile(md, `${transcript.fileName || 'transcript'}.md`, 'text/markdown');
                }}
                className="w-full p-3 bg-zinc-950 hover:bg-zinc-800/80 border border-zinc-800 rounded-2xl text-xs text-left transition flex items-center justify-between group"
              >
                <div className="flex items-center space-x-3">
                  <div className="w-9 h-9 rounded-xl bg-amber-500/20 text-amber-400 flex items-center justify-center text-lg">
                    📝
                  </div>
                  <div>
                    <p className="font-bold text-white">Markdown (.MD)</p>
                    <p className="text-[11px] text-zinc-400">Форматированный отчет по спикерам</p>
                  </div>
                </div>
                <Download className="w-4 h-4 text-zinc-400 group-hover:text-white" />
              </button>

              <button
                onClick={() => downloadFile(JSON.stringify(transcript, null, 2), `${transcript.fileName || 'transcript'}.json`, 'application/json')}
                className="w-full p-3 bg-zinc-950 hover:bg-zinc-800/80 border border-zinc-800 rounded-2xl text-xs text-left transition flex items-center justify-between group"
              >
                <div className="flex items-center space-x-3">
                  <div className="w-9 h-9 rounded-xl bg-cyan-500/20 text-cyan-400 flex items-center justify-center text-lg">
                    🔧
                  </div>
                  <div>
                    <p className="font-bold text-white">JSON структура (.JSON)</p>
                    <p className="text-[11px] text-zinc-400">Сырой массив сегментов и данных</p>
                  </div>
                </div>
                <Download className="w-4 h-4 text-zinc-400 group-hover:text-white" />
              </button>
            </div>
          )}

        </div>

      </div>

    </div>
  );
};
