type ConfidenceGaugeProps = {
  probFake: number;
  verdict: 'REAL' | 'DEEPFAKE';
};

export function ConfidenceGauge({ probFake, verdict }: ConfidenceGaugeProps) {
  const percentage = Math.round(probFake * 100);
  const circumference = 2 * Math.PI * 82;
  const offset = circumference * (1 - probFake);
  const color = verdict === 'DEEPFAKE' ? '#f26b5e' : '#2cc98a';

  return (
    <div className="relative flex min-h-[310px] items-center justify-center overflow-hidden rounded-3xl border border-slate-200 bg-white p-7 shadow-sm">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_45%,rgba(25,169,153,0.10),transparent_58%)]" />
      <svg viewBox="0 0 220 220" className="relative h-60 w-60 -rotate-90">
        <circle cx="110" cy="110" r="82" fill="none" stroke="#e9eff3" strokeWidth="16" />
        <circle cx="110" cy="110" r="82" fill="none" stroke={color} strokeWidth="16" strokeLinecap="round" strokeDasharray={circumference} strokeDashoffset={offset} className="transition-all duration-1000 ease-out" />
      </svg>
      <div className="absolute text-center">
        <div className="text-5xl font-black tracking-tight text-[#0b1f33]">{percentage}%</div>
        <div className="mt-2 text-[11px] font-bold uppercase tracking-[0.2em] text-slate-500">Deepfake confidence</div>
      </div>
      <div className="absolute bottom-5 left-0 right-0 text-center text-xs font-medium text-slate-500">Higher score indicates stronger synthetic signals</div>
    </div>
  );
}
