import { useEffect, useRef, useState } from 'react';
import {
  Activity,
  AlertTriangle,
  AudioLines,
  CheckCircle2,
  ChevronRight,
  CircleHelp,
  FileAudio,
  Fingerprint,
  Info,
  Loader2,
  Mic,
  Pause,
  Play,
  Radio,
  RotateCcw,
  ShieldCheck,
  Sparkles,
  Square,
  UploadCloud,
  Waves,
  X,
  Zap,
} from 'lucide-react';
import { ConfidenceGauge } from '@/components/ConfidenceGauge';
import { SpectrogramCanvas } from '@/components/SpectrogramCanvas';
import { analyzeAudio, decodeAudioFile } from '@/lib/audioAnalysis';
import type { AnalysisResult, ArtifactInfo } from '@/lib/types';

const formatTime = (seconds: number) => `${Math.floor(seconds / 60).toString().padStart(2, '0')}:${Math.floor(seconds % 60).toString().padStart(2, '0')}`;

function App() {
  const [activeView, setActiveView] = useState<'analyze' | 'how'>('analyze');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [activeTab, setActiveTab] = useState<'fingerprint' | 'segments' | 'artifacts'>('fingerprint');
  const [showDetails, setShowDetails] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const recordingChunksRef = useRef<Blob[]>([]);
  const recordingTimerRef = useRef<number | null>(null);

  useEffect(() => () => {
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    if (recordingTimerRef.current) window.clearInterval(recordingTimerRef.current);
  }, [audioUrl]);

  const loadFile = async (file: File) => {
    setError(null);
    setSelectedFile(file);
    setResult(null);
    setIsAnalyzing(true);
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    setAudioUrl(URL.createObjectURL(file));
    try {
      const decoded = await decodeAudioFile(file);
      await new Promise<void>((resolve) => window.setTimeout(resolve, 900));
      setResult(analyzeAudio(decoded.samples, decoded.sampleRate));
    } catch {
      setError('This file could not be decoded. Please try a WAV, MP3, M4A, or FLAC recording.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) void loadFile(file);
  };

  const startRecording = async () => {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      recordingChunksRef.current = [];
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) recordingChunksRef.current.push(event.data);
      };
      recorder.onstop = () => {
        stream.getTracks().forEach((track) => track.stop());
        const blob = new Blob(recordingChunksRef.current, { type: recorder.mimeType || 'audio/webm' });
        const file = new File([blob], `microphone-capture-${Date.now()}.webm`, { type: blob.type });
        void loadFile(file);
      };
      recorder.start();
      mediaRecorderRef.current = recorder;
      setRecordingSeconds(0);
      setIsRecording(true);
      recordingTimerRef.current = window.setInterval(() => setRecordingSeconds((value) => value + 1), 1000);
    } catch {
      setError('Microphone access was not available. Check your browser permission and try again.');
    }
  };

  const stopRecording = () => {
    mediaRecorderRef.current?.stop();
    mediaRecorderRef.current = null;
    setIsRecording(false);
    if (recordingTimerRef.current) window.clearInterval(recordingTimerRef.current);
  };

  const resetAnalysis = () => {
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    setSelectedFile(null);
    setAudioUrl(null);
    setResult(null);
    setError(null);
    setIsPlaying(false);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const togglePlayback = () => {
    setIsPlaying((value) => !value);
  };

  const handleDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragging(false);
    const file = event.dataTransfer.files[0];
    if (file) void loadFile(file);
  };

  return (
    <div className="min-h-screen bg-[#f7fafc] text-[#0b1f33]">
      <header className="border-b border-slate-200/80 bg-white/90 backdrop-blur-xl">
        <div className="mx-auto flex h-[76px] max-w-[1440px] items-center justify-between px-5 lg:px-10">
          <button onClick={() => { setActiveView('analyze'); resetAnalysis(); }} className="flex items-center gap-3 text-left">
            <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-[#0b1f33] text-[#7fe4d4] shadow-lg shadow-slate-300"><Fingerprint size={22} /></span>
            <span><span className="block text-[15px] font-black tracking-tight">SPECTRA<span className="text-[#159c8e]">/TRACE</span></span><span className="block text-[10px] font-bold uppercase tracking-[0.18em] text-slate-400">Audio authenticity lab</span></span>
          </button>
          <nav className="hidden items-center gap-1 rounded-xl bg-slate-100 p-1 md:flex">
            <button onClick={() => setActiveView('analyze')} className={`rounded-lg px-4 py-2 text-sm font-semibold transition ${activeView === 'analyze' ? 'bg-white text-[#0b1f33] shadow-sm' : 'text-slate-500 hover:text-[#0b1f33]'}`}>Analyze audio</button>
            <button onClick={() => setActiveView('how')} className={`rounded-lg px-4 py-2 text-sm font-semibold transition ${activeView === 'how' ? 'bg-white text-[#0b1f33] shadow-sm' : 'text-slate-500 hover:text-[#0b1f33]'}`}>How it works</button>
          </nav>
          <div className="flex items-center gap-3"><span className="hidden items-center gap-2 text-xs font-bold text-slate-500 sm:flex"><span className="h-2 w-2 animate-pulse rounded-full bg-[#27bd82]" /> Browser-based analysis</span><button onClick={() => setActiveView('how')} aria-label="Help" className="rounded-lg p-2 text-slate-400 transition hover:bg-slate-100 hover:text-[#0b1f33]"><CircleHelp size={20} /></button></div>
        </div>
      </header>

      <main className="mx-auto max-w-[1440px] px-5 py-8 lg:px-10 lg:py-12">
        {activeView === 'how' ? <HowItWorks onBack={() => setActiveView('analyze')} /> : (
          <>
            <section className="mb-9 max-w-3xl"><div className="mb-3 flex items-center gap-2 text-xs font-black uppercase tracking-[0.2em] text-[#159c8e]"><Sparkles size={14} /> Novel spectro-temporal fingerprinting</div><h1 className="text-4xl font-black tracking-[-0.045em] text-[#0b1f33] sm:text-5xl lg:text-[4.2rem] lg:leading-[1.02]">Can you hear the <span className="text-[#159c8e]">difference?</span></h1><p className="mt-5 max-w-2xl text-base leading-7 text-slate-500 sm:text-lg">Inspect an audio recording for subtle patterns that often reveal synthetic generation — phase inconsistencies, spectral residue, and unnatural temporal cadence.</p></section>
            <div className="grid gap-7 lg:grid-cols-[minmax(0,1.25fr)_minmax(330px,0.75fr)]">
              <section className="rounded-3xl border border-slate-200 bg-white p-5 shadow-[0_16px_50px_rgba(11,31,51,0.06)] sm:p-7">
                <div className="mb-6 flex items-center justify-between"><div><p className="text-xs font-black uppercase tracking-[0.18em] text-slate-400">Step 01</p><h2 className="mt-1 text-xl font-black tracking-tight">Bring an audio sample</h2></div><div className="flex h-10 w-10 items-center justify-center rounded-xl bg-[#e8f7f4] text-[#159c8e]"><AudioLines size={20} /></div></div>
                {!selectedFile ? <>
                  <div onDragOver={(event) => { event.preventDefault(); setIsDragging(true); }} onDragLeave={() => setIsDragging(false)} onDrop={handleDrop} onClick={() => fileInputRef.current?.click()} className={`group cursor-pointer rounded-2xl border-2 border-dashed px-6 py-12 text-center transition sm:py-16 ${isDragging ? 'border-[#159c8e] bg-[#effbf8]' : 'border-slate-200 bg-slate-50/70 hover:border-[#8acfc5] hover:bg-[#f4fbfa]'}`}><input ref={fileInputRef} type="file" accept="audio/*,.flac" onChange={handleFileChange} className="hidden" /><div className="mx-auto flex h-16 w-16 items-center justify-center rounded-2xl bg-white text-[#159c8e] shadow-sm transition group-hover:-translate-y-1"><UploadCloud size={28} /></div><h3 className="mt-5 font-bold text-[#0b1f33]">Drop your recording here</h3><p className="mt-2 text-sm text-slate-400">or click to browse from your device</p><div className="mt-5 flex justify-center gap-2"><span className="rounded-md bg-white px-2.5 py-1 text-[10px] font-black uppercase tracking-wider text-slate-400 shadow-sm">WAV</span><span className="rounded-md bg-white px-2.5 py-1 text-[10px] font-black uppercase tracking-wider text-slate-400 shadow-sm">MP3</span><span className="rounded-md bg-white px-2.5 py-1 text-[10px] font-black uppercase tracking-wider text-slate-400 shadow-sm">FLAC</span></div></div>
                  <div className="my-5 flex items-center gap-4"><div className="h-px flex-1 bg-slate-200" /><span className="text-xs font-bold uppercase tracking-widest text-slate-400">or</span><div className="h-px flex-1 bg-slate-200" /></div>
                  <button onClick={startRecording} className="flex w-full items-center justify-center gap-2 rounded-xl border border-slate-200 bg-white px-4 py-3.5 text-sm font-bold text-[#0b1f33] transition hover:border-[#8acfc5] hover:bg-[#f4fbfa]"><Mic size={17} className="text-[#159c8e]" /> Record from microphone</button>
                </> : <AudioPreview file={selectedFile} audioUrl={audioUrl} isPlaying={isPlaying} isRecording={isRecording} recordingSeconds={recordingSeconds} onTogglePlayback={togglePlayback} onReset={resetAnalysis} onStopRecording={stopRecording} />}
                {isRecording && <div className="mt-4 flex items-center justify-between rounded-xl border border-red-100 bg-red-50 px-4 py-3 text-sm font-bold text-red-600"><span className="flex items-center gap-2"><span className="h-2 w-2 animate-pulse rounded-full bg-red-500" /> Recording in progress</span><button onClick={stopRecording} className="rounded-lg bg-red-500 px-3 py-1.5 text-xs font-bold text-white hover:bg-red-600">Stop & analyze</button></div>}
                {error && <div className="mt-4 flex items-start gap-3 rounded-xl border border-red-100 bg-red-50 p-4 text-sm text-red-700"><AlertTriangle size={18} className="mt-0.5 shrink-0" /><span>{error}</span><button onClick={() => setError(null)} className="ml-auto"><X size={16} /></button></div>}
              </section>
              <PipelineCard />
            </div>

            {isAnalyzing && <div className="mt-8 flex items-center gap-4 rounded-2xl border border-[#bfe9e2] bg-[#effbf8] p-5"><Loader2 size={22} className="animate-spin text-[#159c8e]" /><div><p className="font-bold text-[#0b1f33]">Building your audio fingerprint</p><p className="mt-1 text-sm text-slate-500">Mapping spectral energy, phase coherence, and temporal patterns...</p></div></div>}
            {result && !isAnalyzing && <Results result={result} activeTab={activeTab} setActiveTab={setActiveTab} showDetails={showDetails} setShowDetails={setShowDetails} />}
          </>
        )}
      </main>
      <footer className="border-t border-slate-200 bg-white"><div className="mx-auto flex max-w-[1440px] flex-col justify-between gap-3 px-5 py-6 text-xs text-slate-400 sm:flex-row sm:items-center lg:px-10"><span>Built for responsible audio authenticity research.</span><span className="flex items-center gap-2"><ShieldCheck size={14} className="text-[#159c8e]" /> No files leave your browser</span></div></footer>
    </div>
  );
}

function AudioPreview({ file, audioUrl, isPlaying, isRecording, recordingSeconds, onTogglePlayback, onReset, onStopRecording }: { file: File; audioUrl: string | null; isPlaying: boolean; isRecording: boolean; recordingSeconds: number; onTogglePlayback: () => void; onReset: () => void; onStopRecording: () => void }) {
  const audioElementRef = useRef<HTMLAudioElement | null>(null);
  const togglePlayback = () => {
    if (!audioElementRef.current) return;
    if (isPlaying) void audioElementRef.current.pause();
    else void audioElementRef.current.play();
    onTogglePlayback();
  };
  return <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4"><div className="flex items-center gap-4"><button onClick={isRecording ? onStopRecording : togglePlayback} className={`flex h-12 w-12 shrink-0 items-center justify-center rounded-full text-white shadow-lg transition hover:scale-105 ${isRecording ? 'bg-red-500 shadow-red-200' : 'bg-[#0b1f33] shadow-slate-200'}`}>{isRecording ? <Square size={17} fill="currentColor" /> : isPlaying ? <Pause size={19} fill="currentColor" /> : <Play size={19} fill="currentColor" className="ml-0.5" />}</button><div className="min-w-0 flex-1"><p className="truncate text-sm font-bold text-[#0b1f33]">{file.name}</p><p className="mt-1 text-xs text-slate-400">{isRecording ? `Recording ${formatTime(recordingSeconds)}` : `${(file.size / 1024 / 1024).toFixed(2)} MB · Ready for analysis`}</p><div className="mt-3 flex h-5 items-end gap-0.5">{Array.from({ length: 48 }, (_, index) => <span key={index} className={`w-1 rounded-full ${isRecording ? 'animate-pulse bg-red-400' : 'bg-[#6dcfc1]'}`} style={{ height: `${25 + ((index * 17) % 65)}%`, animationDelay: `${index * 20}ms` }} />)}</div></div><button onClick={onReset} aria-label="Remove audio" className="rounded-lg p-2 text-slate-400 hover:bg-white hover:text-red-500"><X size={18} /></button></div>{audioUrl && !isRecording && <audio ref={audioElementRef} src={audioUrl} className="hidden" onEnded={onTogglePlayback} />}</div>;
}

function PipelineCard() {
  const steps = [{ icon: Waves, title: 'Normalize', text: '16 kHz mono, silence trim' }, { icon: Activity, title: 'Fingerprint', text: 'Mel, STFT & phase maps' }, { icon: Zap, title: 'Detect', text: 'Dual-branch AI classifier' }];
  return <section className="rounded-3xl bg-[#0b1f33] p-6 text-white shadow-[0_16px_50px_rgba(11,31,51,0.15)] sm:p-7"><div className="mb-8 flex items-center justify-between"><div><p className="text-xs font-black uppercase tracking-[0.18em] text-[#7fe4d4]">Step 02</p><h2 className="mt-1 text-xl font-black tracking-tight">Signal intelligence</h2></div><Radio className="text-[#7fe4d4]" size={22} /></div><div className="space-y-6">{steps.map(({ icon: Icon, title, text }, index) => <div key={title} className="relative flex gap-4">{index < steps.length - 1 && <div className="absolute left-4 top-9 h-7 w-px bg-white/15" />}<div className="z-10 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-white/10 text-[#7fe4d4]"><Icon size={16} /></div><div><h3 className="text-sm font-bold">{title}</h3><p className="mt-1 text-xs leading-5 text-slate-400">{text}</p></div></div>)}</div><div className="mt-9 rounded-xl border border-white/10 bg-white/5 p-4"><div className="flex items-center gap-2 text-xs font-bold text-[#b5ebe3]"><Info size={14} /> What are we looking for?</div><p className="mt-2 text-xs leading-5 text-slate-400">Synthetic audio often leaves a measurable trace in the relationship between frequency energy and phase over time.</p></div></section>;
}

function Results({ result, activeTab, setActiveTab, showDetails, setShowDetails }: { result: AnalysisResult; activeTab: 'fingerprint' | 'segments' | 'artifacts'; setActiveTab: (tab: 'fingerprint' | 'segments' | 'artifacts') => void; showDetails: boolean; setShowDetails: (value: boolean) => void }) {
  const isFake = result.verdict === 'DEEPFAKE';
  return <section className="mt-12 border-t border-slate-200 pt-10"><div className="mb-7 flex flex-col justify-between gap-4 sm:flex-row sm:items-end"><div><p className="text-xs font-black uppercase tracking-[0.18em] text-slate-400">Analysis complete</p><h2 className="mt-2 text-3xl font-black tracking-tight">Your audio fingerprint</h2><p className="mt-2 text-sm text-slate-500">A multi-signal view of what the detector found in this recording.</p></div><button onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })} className="flex items-center gap-2 self-start rounded-xl border border-slate-200 bg-white px-4 py-2.5 text-sm font-bold text-[#0b1f33] shadow-sm transition hover:border-[#8acfc5]"><RotateCcw size={15} /> Analyze another</button></div><div className="grid gap-6 lg:grid-cols-[1fr_1fr]"><div className={`rounded-3xl border p-7 ${isFake ? 'border-red-100 bg-[#fff8f7]' : 'border-emerald-100 bg-[#f4fcf8]'}`}><div className="flex items-start justify-between"><div><div className={`flex items-center gap-2 text-xs font-black uppercase tracking-[0.18em] ${isFake ? 'text-red-500' : 'text-emerald-600'}`}>{isFake ? <AlertTriangle size={15} /> : <CheckCircle2 size={15} />} Model verdict</div><div className="mt-3 text-4xl font-black tracking-tight text-[#0b1f33]">{result.verdict}</div><p className="mt-2 max-w-md text-sm leading-6 text-slate-500">{isFake ? 'Synthetic signals were detected across the analyzed spectro-temporal fingerprint.' : 'No strong synthetic signature was detected in the analyzed spectro-temporal fingerprint.'}</p></div><div className={`rounded-2xl p-3 ${isFake ? 'bg-red-100 text-red-500' : 'bg-emerald-100 text-emerald-600'}`}>{isFake ? <AlertTriangle size={24} /> : <ShieldCheck size={24} />}</div></div><div className="mt-8 grid grid-cols-3 gap-3"><Metric label="Deepfake" value={`${Math.round(result.probFake * 100)}%`} /><Metric label="Real" value={`${Math.round(result.probReal * 100)}%`} /><Metric label="Duration" value={`${result.durationSec.toFixed(1)}s`} /></div></div><ConfidenceGauge probFake={result.probFake} verdict={result.verdict} /></div><div className="mt-7 flex flex-wrap gap-2 border-b border-slate-200 pb-3">{([['fingerprint', 'Fingerprint maps'], ['segments', 'Segment confidence'], ['artifacts', 'Artifact notes']] as const).map(([tab, label]) => <button key={tab} onClick={() => setActiveTab(tab)} className={`rounded-lg px-4 py-2.5 text-sm font-bold transition ${activeTab === tab ? 'bg-[#0b1f33] text-white' : 'text-slate-500 hover:bg-slate-100 hover:text-[#0b1f33]'}`}>{label}</button>)}</div>{activeTab === 'fingerprint' && <div className="mt-6 grid gap-5 xl:grid-cols-3"><SpectrogramCanvas data={result.melSpectrogram} title="Mel-spectrogram" subtitle="Frequency-energy distribution" palette="ember" /><SpectrogramCanvas data={result.stftMagnitude} title="STFT magnitude" subtitle="Spectral texture and residue" palette="ocean" /><SpectrogramCanvas data={result.phasePattern} title="Phase pattern" subtitle="Instantaneous phase behavior" palette="phase" /></div>}{activeTab === 'segments' && <SegmentTable result={result} />}{activeTab === 'artifacts' && <ArtifactPanel artifacts={result.artifacts} isFake={isFake} />}{activeTab === 'fingerprint' && <div className="mt-8 rounded-2xl border border-slate-200 bg-white p-5"><button onClick={() => setShowDetails(!showDetails)} className="flex w-full items-center justify-between text-left"><span className="flex items-center gap-3 text-sm font-bold"><Fingerprint size={18} className="text-[#159c8e]" /> Channel attention weights</span><ChevronRight size={18} className={`text-slate-400 transition ${showDetails ? 'rotate-90' : ''}`} /></button>{showDetails && <div className="mt-5 grid gap-3 sm:grid-cols-5">{result.attentionWeights.map((weight, index) => <div key={index}><div className="mb-1 flex justify-between text-[10px] font-bold uppercase tracking-wider text-slate-400"><span>Channel {index + 1}</span><span>{Math.round(weight * 100)}%</span></div><div className="h-2 overflow-hidden rounded-full bg-slate-100"><div className="h-full rounded-full bg-[#159c8e]" style={{ width: `${weight * 100}%` }} /></div></div>)}</div>}</div>}</section>;
}

function SegmentTable({ result }: { result: AnalysisResult }) { return <div className="mt-6 overflow-hidden rounded-2xl border border-slate-200 bg-white"><div className="grid grid-cols-[1.3fr_1fr_1fr] border-b border-slate-200 bg-slate-50 px-5 py-3 text-[10px] font-black uppercase tracking-[0.14em] text-slate-400"><span>Analysis window</span><span>Verdict</span><span>Deepfake confidence</span></div>{result.segments.map((segment) => <div key={segment.index} className="grid grid-cols-[1.3fr_1fr_1fr] items-center border-b border-slate-100 px-5 py-4 text-sm last:border-0"><span className="flex items-center gap-3 font-semibold text-[#0b1f33]"><span className="flex h-7 w-7 items-center justify-center rounded-lg bg-slate-100 text-xs font-black text-slate-500">{segment.index}</span>{formatTime(segment.startTimeSec)} — {formatTime(segment.endTimeSec)}</span><span className={`font-bold ${segment.verdict === 'DEEPFAKE' ? 'text-red-500' : 'text-emerald-600'}`}>{segment.verdict}</span><span className="flex items-center gap-3 font-bold text-slate-600"><span className="h-2 max-w-32 flex-1 overflow-hidden rounded-full bg-slate-100"><span className={`block h-full rounded-full ${segment.verdict === 'DEEPFAKE' ? 'bg-red-400' : 'bg-emerald-400'}`} style={{ width: `${segment.probFake * 100}%` }} /></span>{Math.round(segment.probFake * 100)}%</span></div>)}</div>; }

function ArtifactPanel({ artifacts, isFake }: { artifacts: ArtifactInfo[]; isFake: boolean }) { return <div className="mt-6 grid gap-4 md:grid-cols-3">{artifacts.map((artifact) => <div key={artifact.name} className="rounded-2xl border border-slate-200 bg-white p-5"><div className="flex items-start justify-between"><div className={`rounded-xl p-2.5 ${artifact.severity === 'high' ? 'bg-red-50 text-red-500' : artifact.severity === 'moderate' ? 'bg-amber-50 text-amber-500' : 'bg-emerald-50 text-emerald-500'}`}>{artifact.severity === 'high' ? <AlertTriangle size={18} /> : <Activity size={18} />}</div><span className={`rounded-full px-2.5 py-1 text-[10px] font-black uppercase tracking-wider ${artifact.severity === 'high' ? 'bg-red-50 text-red-500' : artifact.severity === 'moderate' ? 'bg-amber-50 text-amber-600' : 'bg-emerald-50 text-emerald-600'}`}>{artifact.severity}</span></div><h3 className="mt-5 font-bold text-[#0b1f33]">{artifact.name}</h3><p className="mt-2 text-sm leading-6 text-slate-500">{artifact.description}</p><div className="mt-5 flex items-center justify-between text-xs font-bold text-slate-400"><span>Signal confidence</span><span className="text-[#0b1f33]">{Math.round(artifact.confidence * 100)}%</span></div><div className="mt-2 h-1.5 overflow-hidden rounded-full bg-slate-100"><div className={`h-full rounded-full ${isFake ? 'bg-red-400' : 'bg-[#159c8e]'}`} style={{ width: `${artifact.confidence * 100}%` }} /></div></div>)}</div>; }

function Metric({ label, value }: { label: string; value: string }) { return <div className="rounded-xl bg-white/70 p-3"><div className="text-[10px] font-black uppercase tracking-wider text-slate-400">{label}</div><div className="mt-1 text-lg font-black text-[#0b1f33]">{value}</div></div>; }

function HowItWorks({ onBack }: { onBack: () => void }) { const items = [{ icon: FileAudio, title: 'Audio normalization', text: 'Every sample is converted to a consistent 16 kHz mono representation. Silence is trimmed and loudness is balanced so recording volume does not bias the result.' }, { icon: Waves, title: 'Spectro-temporal fingerprint', text: 'The detector reads three complementary views: Mel energy, STFT magnitude, and phase behavior. Together they reveal patterns that are easy to miss in a waveform.' }, { icon: Activity, title: 'Dual-branch detection', text: 'A spectral branch learns frequency artifacts while a temporal branch studies cadence, pauses, and frame-to-frame behavior. An attention layer highlights the strongest evidence.' }, { icon: ShieldCheck, title: 'Responsible interpretation', text: 'The result is an evidence-based screening signal, not proof of fraud. High-impact decisions should always include provenance checks and human review.' }]; return <section className="mx-auto max-w-4xl"><button onClick={onBack} className="mb-9 flex items-center gap-2 text-sm font-bold text-[#159c8e] hover:text-[#0b1f33]"><ChevronRight size={16} className="rotate-180" /> Back to analyzer</button><div className="mb-12"><p className="text-xs font-black uppercase tracking-[0.18em] text-[#159c8e]">Inside the signal</p><h1 className="mt-3 text-4xl font-black tracking-tight sm:text-5xl">How Spectra/Trace listens</h1><p className="mt-5 max-w-2xl text-lg leading-8 text-slate-500">Modern voice generators can sound remarkably natural. The key is not just what a voice says, but how its frequency and phase evolve over time.</p></div><div className="space-y-4">{items.map(({ icon: Icon, title, text }, index) => <div key={title} className="flex gap-5 rounded-2xl border border-slate-200 bg-white p-6 shadow-sm"><div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-xl bg-[#e8f7f4] text-[#159c8e]"><Icon size={22} /></div><div><div className="flex items-center gap-3"><span className="text-xs font-black text-slate-300">0{index + 1}</span><h2 className="text-lg font-black">{title}</h2></div><p className="mt-2 text-sm leading-7 text-slate-500">{text}</p></div></div>)}</div></section>; }

export default App;
