import React, { useState, useEffect } from 'react';
import { AlertCircle, X } from 'lucide-react';
import { Navbar } from './components/Navbar';
import { UploadSection } from './components/UploadSection';
import { TranscriptView } from './components/TranscriptView';
import { HistoryDrawer } from './components/HistoryDrawer';
import { InstructionsModal } from './components/InstructionsModal';
import { TranscriptRecord, TranscriptionOptions } from './types';

export default function App() {
  const [activeTab, setActiveTab] = useState<'transcribe' | 'history' | 'guide'>('transcribe');
  const [savedRecords, setSavedRecords] = useState<TranscriptRecord[]>([]);
  const [currentTranscript, setCurrentTranscript] = useState<TranscriptRecord | null>(null);

  // Processing state
  const [isProcessing, setIsProcessing] = useState(false);
  const [processStep, setProcessStep] = useState('');
  const [processPercent, setProcessPercent] = useState(0);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Load history from localStorage
  useEffect(() => {
    try {
      const stored = localStorage.getItem('tranzip_history');
      if (stored) {
        setSavedRecords(JSON.parse(stored));
      }
    } catch (e) {
      console.warn('Could not read saved history from localStorage');
    }
  }, []);

  // Save history helper
  const saveToHistory = (newRecord: TranscriptRecord) => {
    const updated = [newRecord, ...savedRecords.filter((r) => r.id !== newRecord.id)];
    setSavedRecords(updated);
    try {
      localStorage.setItem('tranzip_history', JSON.stringify(updated));
    } catch (e) {
      console.warn('Could not save to localStorage');
    }
  };

  const handleUpdateTranscript = (updated: TranscriptRecord) => {
    setCurrentTranscript(updated);
    saveToHistory(updated);
  };

  const handleDeleteRecord = (id: string) => {
    const updated = savedRecords.filter((r) => r.id !== id);
    setSavedRecords(updated);
    try {
      localStorage.setItem('tranzip_history', JSON.stringify(updated));
    } catch (e) {}
    if (currentTranscript?.id === id) {
      setCurrentTranscript(null);
    }
  };

  const handleClearHistory = () => {
    if (window.confirm('Вы действительно хотите очистить историю расшифровок?')) {
      setSavedRecords([]);
      localStorage.removeItem('tranzip_history');
      setCurrentTranscript(null);
    }
  };

  // Main Transcription Trigger
  const handleStartTranscription = async (payload: {
    fileData?: string;
    chunks?: any[];
    mimeType: string;
    fileName: string;
    fileType: 'audio' | 'video';
    mediaUrl?: string;
    options: TranscriptionOptions;
    sampleId?: string;
  }) => {
    setIsProcessing(true);
    setErrorMessage(null);
    setProcessPercent(10);
    setProcessStep('Инициализация медиафайла...');

    // Progress step simulation
    const interval = setInterval(() => {
      setProcessPercent((prev) => {
        if (prev < 40) {
          setProcessStep('Анализ медиапотока и частот...');
          return prev + 10;
        } else if (prev < 80) {
          setProcessStep('Нейросетевая расшифровка Gemini 2.5 AI...');
          return prev + 15;
        } else if (prev < 95) {
          setProcessStep('Формирование субтитров и спикер-диаризации...');
          return prev + 5;
        }
        return prev;
      });
    }, 400);

    try {
      const res = await fetch('/api/transcribe', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      clearInterval(interval);
      setProcessPercent(100);
      setProcessStep('Готово!');

      if (!res.ok) {
        let errorMsg = 'Ошибка сервера при обработке файла';
        try {
          const errBody = await res.json();
          errorMsg = errBody.error || errorMsg;
        } catch (e) {
          const rawText = await res.text();
          errorMsg = `Ошибка ${res.status}: ${rawText.slice(0, 100)}`;
        }
        throw new Error(errorMsg);
      }

      const data = await res.json();

      if (data.success && data.data) {
        const recordData = data.data;
        const newRecord: TranscriptRecord = {
          id: `tr-${Date.now()}`,
          title: payload.fileName,
          fileName: payload.fileName,
          fileType: payload.fileType,
          fileSize: 0,
          durationSec: recordData.durationEstimateSec || 45,
          mediaUrl: payload.mediaUrl,
          mimeType: payload.mimeType,
          createdAt: new Date().toISOString(),
          languageDetected: recordData.languageDetected || 'Русский',
          segments: recordData.segments || [],
          summary: recordData.summary || { overview: '', keyPoints: [], actionItems: [] },
          srtContent: recordData.srtContent || '',
          rawText: recordData.rawText || '',
          wordCount: recordData.wordCount || 0,
        };

        setCurrentTranscript(newRecord);
        saveToHistory(newRecord);
      } else {
        setErrorMessage(data.error || 'Ошибка при расшифровке медиафайла');
      }
    } catch (err: any) {
      clearInterval(interval);
      setErrorMessage(err.message || 'Произошла ошибка при обработке медиафайла.');
    } finally {
      setIsProcessing(false);
      setProcessPercent(0);
      setProcessStep('');
    }
  };

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 flex flex-col font-sans selection:bg-indigo-500 selection:text-white relative overflow-hidden">
      
      {/* Radial Background Accent Glow */}
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-full max-w-7xl h-[500px] bg-[radial-gradient(ellipse_80%_80%_at_50%_-20%,rgba(99,102,241,0.15),rgba(255,255,255,0))] pointer-events-none" />

      {/* Top Navbar */}
      <Navbar
        activeTab={activeTab}
        setActiveTab={(tab) => {
          setActiveTab(tab);
        }}
        savedCount={savedRecords.length}
      />

      {/* Main Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6 sm:py-8 space-y-6 relative z-10">
        
        {/* Error Banner */}
        {errorMessage && (
          <div className="bg-red-500/10 border border-red-500/30 text-red-300 px-4 py-3 rounded-2xl flex items-start justify-between gap-3 shadow-lg animate-in fade-in slide-in-from-top-2">
            <div className="flex items-start space-x-3">
              <AlertCircle className="w-5 h-5 text-red-400 mt-0.5 shrink-0" />
              <div>
                <p className="font-medium text-sm text-red-200">Ошибка расшифровки</p>
                <p className="text-xs text-red-300/80 mt-0.5">{errorMessage}</p>
              </div>
            </div>
            <button
              onClick={() => setErrorMessage(null)}
              className="text-red-400 hover:text-red-200 p-1 rounded-lg hover:bg-red-500/20 transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        )}
        
        {/* TAB 1: Transcribe View */}
        {activeTab === 'transcribe' && (
          <div>
            {!currentTranscript ? (
              <UploadSection
                onStartTranscription={handleStartTranscription}
                isProcessing={isProcessing}
                processStep={processStep}
                processPercent={processPercent}
              />
            ) : (
              <TranscriptView
                transcript={currentTranscript}
                onUpdateTranscript={handleUpdateTranscript}
                onNewTranscription={() => setCurrentTranscript(null)}
              />
            )}
          </div>
        )}

        {/* TAB 2: History View */}
        {activeTab === 'history' && (
          <HistoryDrawer
            records={savedRecords}
            onSelectRecord={(rec) => {
              setCurrentTranscript(rec);
              setActiveTab('transcribe');
            }}
            onDeleteRecord={handleDeleteRecord}
            onClearHistory={handleClearHistory}
          />
        )}

        {/* TAB 3: Guide View */}
        {activeTab === 'guide' && <InstructionsModal />}

      </main>

      {/* Footer */}
      <footer className="border-t border-zinc-800/80 bg-zinc-950 py-6 text-center text-xs text-zinc-500 relative z-10">
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-3">
          <p>© 2026 Tranzip AI. Сервис онлайн-транскрибации речи и субтитров.</p>
          <div className="flex items-center space-x-4 text-zinc-400">
            <span>Обработка Gemini 2.5 Flash</span>
            <span>•</span>
            <span>Поддержка 30+ языков</span>
          </div>
        </div>
      </footer>

    </div>
  );
}
