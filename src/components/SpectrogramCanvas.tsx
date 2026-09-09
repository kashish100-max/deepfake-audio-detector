import { useEffect, useRef } from 'react';

type SpectrogramCanvasProps = {
  data: number[][];
  title: string;
  subtitle: string;
  palette: 'ember' | 'ocean' | 'phase';
};

const palettes: Record<SpectrogramCanvasProps['palette'], (value: number) => [number, number, number]> = {
  ember: (value) => [Math.round(25 + value * 225), Math.round(23 + value * 92), Math.round(32 + value * 22)],
  ocean: (value) => [Math.round(10 + value * 30), Math.round(45 + value * 160), Math.round(70 + value * 165)],
  phase: (value) => {
    const red = Math.round(22 + value * 215);
    const blue = Math.round(215 - value * 165);
    const green = Math.round(80 + (1 - Math.abs(value - 0.5) * 2) * 80);
    return [red, green, blue];
  },
};

export function SpectrogramCanvas({ data, title, subtitle, palette }: SpectrogramCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !data.length || !data[0]?.length) return;
    const context = canvas.getContext('2d');
    if (!context) return;
    const pixelRatio = window.devicePixelRatio || 1;
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    canvas.width = width * pixelRatio;
    canvas.height = height * pixelRatio;
    context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
    const cellWidth = width / data[0].length;
    const cellHeight = height / data.length;
    const color = palettes[palette];
    data.forEach((row, rowIndex) => {
      row.forEach((value, columnIndex) => {
        const [red, green, blue] = color(Math.max(0, Math.min(1, value)));
        context.fillStyle = `rgb(${red}, ${green}, ${blue})`;
        context.fillRect(columnIndex * cellWidth, height - (rowIndex + 1) * cellHeight, cellWidth + 0.75, cellHeight + 0.75);
      });
    });
  }, [data, palette]);

  return (
    <div className="overflow-hidden rounded-2xl border border-slate-200 bg-[#0b1f33] shadow-sm">
      <div className="flex items-center justify-between border-b border-white/10 px-5 py-4">
        <div>
          <h3 className="font-semibold text-white">{title}</h3>
          <p className="mt-1 text-xs text-slate-300">{subtitle}</p>
        </div>
        <span className="rounded-full border border-white/15 bg-white/10 px-3 py-1 text-[10px] font-bold uppercase tracking-[0.16em] text-cyan-100">64 × 150</span>
      </div>
      <div className="p-4">
        <div className="relative h-64 w-full overflow-hidden rounded-xl bg-slate-950">
          <canvas ref={canvasRef} className="h-full w-full" />
          <div className="pointer-events-none absolute inset-x-3 bottom-2 flex justify-between text-[10px] font-medium text-white/60"><span>0.0s</span><span>1.5s</span><span>3.0s</span></div>
          <div className="pointer-events-none absolute inset-y-3 left-2 flex flex-col justify-between text-[10px] font-medium text-white/60"><span>high</span><span>mid</span><span>low</span></div>
        </div>
        <div className="mt-3 flex items-center justify-between text-[10px] uppercase tracking-[0.14em] text-slate-400"><span>Frequency bands</span><span>Time →</span></div>
      </div>
    </div>
  );
}
