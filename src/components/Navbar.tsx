import React from 'react';
import { Mic, FileText, History, HelpCircle, Sparkles } from 'lucide-react';

interface NavbarProps {
  activeTab: 'transcribe' | 'history' | 'guide';
  setActiveTab: (tab: 'transcribe' | 'history' | 'guide') => void;
  savedCount: number;
}

export const Navbar: React.FC<NavbarProps> = ({ activeTab, setActiveTab, savedCount }) => {
  return (
    <header className="sticky top-0 z-50 pt-3 pb-2 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto w-full">
      <div className="bg-zinc-900/80 backdrop-blur-xl border border-zinc-800/90 rounded-2xl sm:rounded-3xl px-4 sm:px-6 h-16 flex items-center justify-between shadow-2xl shadow-black/50">
        
        {/* Brand Logo & Model Tag */}
        <div 
          className="flex items-center space-x-3 cursor-pointer group" 
          onClick={() => setActiveTab('transcribe')}
        >
          <div className="w-10 h-10 rounded-2xl bg-gradient-to-tr from-indigo-600 via-indigo-500 to-emerald-400 flex items-center justify-center shadow-lg shadow-indigo-600/30 group-hover:scale-105 transition-transform">
            <Mic className="w-5 h-5 text-white" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <span className="font-extrabold text-lg tracking-tight text-white">Tranzip</span>
              <span className="text-[10px] uppercase font-bold px-2.5 py-0.5 rounded-full bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 flex items-center gap-1">
                <Sparkles className="w-3 h-3 text-indigo-400" /> Gemini 3.6
              </span>
            </div>
            <p className="text-[11px] text-zinc-400 hidden sm:block">AI-транскрибация аудио и видео</p>
          </div>
        </div>

        {/* Navigation Pills */}
        <nav className="flex items-center space-x-1 sm:space-x-2 bg-zinc-950/80 p-1 rounded-2xl border border-zinc-800/80">
          <button
            onClick={() => setActiveTab('transcribe')}
            className={`px-3.5 sm:px-4 py-1.5 rounded-xl text-xs sm:text-sm font-semibold transition-all flex items-center gap-2 ${
              activeTab === 'transcribe'
                ? 'bg-zinc-100 text-zinc-950 shadow-md font-bold'
                : 'text-zinc-400 hover:text-white hover:bg-zinc-800/60'
            }`}
          >
            <FileText className="w-4 h-4" />
            <span>Расшифровка</span>
          </button>

          <button
            onClick={() => setActiveTab('history')}
            className={`px-3.5 sm:px-4 py-1.5 rounded-xl text-xs sm:text-sm font-semibold transition-all flex items-center gap-2 relative ${
              activeTab === 'history'
                ? 'bg-zinc-100 text-zinc-950 shadow-md font-bold'
                : 'text-zinc-400 hover:text-white hover:bg-zinc-800/60'
            }`}
          >
            <History className="w-4 h-4" />
            <span>История</span>
            {savedCount > 0 && (
              <span className="ml-0.5 px-1.5 py-0.2 text-[10px] rounded-full bg-indigo-500/20 text-indigo-300 font-bold border border-indigo-500/30">
                {savedCount}
              </span>
            )}
          </button>

          <button
            onClick={() => setActiveTab('guide')}
            className={`px-3.5 sm:px-4 py-1.5 rounded-xl text-xs sm:text-sm font-semibold transition-all flex items-center gap-2 ${
              activeTab === 'guide'
                ? 'bg-zinc-100 text-zinc-950 shadow-md font-bold'
                : 'text-zinc-400 hover:text-white hover:bg-zinc-800/60'
            }`}
          >
            <HelpCircle className="w-4 h-4" />
            <span className="hidden md:inline">Инструкция</span>
          </button>
        </nav>

      </div>
    </header>
  );
};
