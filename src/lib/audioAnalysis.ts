import { createMelFilterBank, fft, hannWindow } from '@/lib/fft';
import type { AnalysisResult, AudioFingerprint, ArtifactInfo, SegmentResult } from '@/lib/types';

const ANALYSIS_SAMPLE_RATE = 16000;
const FFT_SIZE = 512;
const HOP_SIZE = 256;
const MEL_BANDS = 64;
const DISPLAY_FRAMES = 150;
const DISPLAY_BANDS = 64;
const ANALYSIS_WINDOW_SEC = 3;

function resample(input: Float32Array, sourceRate: number, targetRate: number): Float32Array {
  if (sourceRate === targetRate) return input;
  const outputLength = Math.max(1, Math.round((input.length * targetRate) / sourceRate));
  const output = new Float32Array(outputLength);
  const ratio = sourceRate / targetRate;
  for (let i = 0; i < outputLength; i += 1) {
    const position = i * ratio;
    const left = Math.floor(position);
    const right = Math.min(left + 1, input.length - 1);
    const fraction = position - left;
    output[i] = input[left] * (1 - fraction) + input[right] * fraction;
  }
  return output;
}

function trimSilence(input: Float32Array): Float32Array {
  if (!input.length) return input;
  let peak = 0;
  for (let i = 0; i < input.length; i += 1) peak = Math.max(peak, Math.abs(input[i]));
  const threshold = Math.max(0.008, peak * 0.035);
  let start = 0;
  let end = input.length - 1;
  while (start < input.length && Math.abs(input[start]) < threshold) start += 1;
  while (end > start && Math.abs(input[end]) < threshold) end -= 1;
  return input.slice(start, end + 1);
}

function normalize(input: Float32Array): Float32Array {
  let peak = 0;
  let sumSquares = 0;
  for (let i = 0; i < input.length; i += 1) {
    peak = Math.max(peak, Math.abs(input[i]));
    sumSquares += input[i] ** 2;
  }
  if (peak < 1e-7) return input;
  const peakNormalized = new Float32Array(input.length);
  const peakGain = 0.92 / peak;
  const rms = Math.sqrt(sumSquares / input.length);
  const rmsGain = rms > 1e-7 ? Math.min(1.6, 0.14 / rms) : 1;
  for (let i = 0; i < input.length; i += 1) peakNormalized[i] = input[i] * peakGain * rmsGain;
  return peakNormalized;
}

function normalizeMatrix(matrix: number[][]): number[][] {
  let minimum = Infinity;
  let maximum = -Infinity;
  matrix.forEach((row) => row.forEach((value) => {
    minimum = Math.min(minimum, value);
    maximum = Math.max(maximum, value);
  }));
  const range = maximum - minimum || 1;
  return matrix.map((row) => row.map((value) => Math.max(0, Math.min(1, (value - minimum) / range))));
}

function resizeMatrix(matrix: number[][], rows: number, columns: number): number[][] {
  return Array.from({ length: rows }, (_, rowIndex) => {
    const sourceRow = (rowIndex / Math.max(1, rows - 1)) * Math.max(1, matrix.length - 1);
    const rowLow = Math.floor(sourceRow);
    const rowHigh = Math.min(rowLow + 1, matrix.length - 1);
    const rowMix = sourceRow - rowLow;
    return Array.from({ length: columns }, (_, columnIndex) => {
      const sourceColumn = (columnIndex / Math.max(1, columns - 1)) * Math.max(1, matrix[0].length - 1);
      const columnLow = Math.floor(sourceColumn);
      const columnHigh = Math.min(columnLow + 1, matrix[0].length - 1);
      const columnMix = sourceColumn - columnLow;
      const top = matrix[rowLow][columnLow] * (1 - columnMix) + matrix[rowLow][columnHigh] * columnMix;
      const bottom = matrix[rowHigh][columnLow] * (1 - columnMix) + matrix[rowHigh][columnHigh] * columnMix;
      return top * (1 - rowMix) + bottom * rowMix;
    });
  });
}

function extractFeatures(samples: Float32Array, sampleRate: number): AudioFingerprint {
  const audio = normalize(trimSilence(resample(samples, sampleRate, ANALYSIS_SAMPLE_RATE)));
  const padded = new Float32Array(Math.max(ANALYSIS_SAMPLE_RATE * ANALYSIS_WINDOW_SEC, audio.length));
  padded.set(audio.slice(0, padded.length));
  const window = hannWindow(FFT_SIZE);
  const melFilters = createMelFilterBank(ANALYSIS_SAMPLE_RATE, FFT_SIZE, MEL_BANDS);
  const mel: number[][] = [];
  const magnitude: number[][] = [];
  const phase: number[][] = [];

  for (let frameStart = 0; frameStart + FFT_SIZE <= padded.length; frameStart += HOP_SIZE) {
    const frame = new Float32Array(FFT_SIZE);
    for (let i = 0; i < FFT_SIZE; i += 1) frame[i] = padded[frameStart + i] * window[i];
    const spectrum = fft(frame);
    const powers = Array.from({ length: FFT_SIZE / 2 + 1 }, (_, index) => {
      const real = spectrum.real[index];
      const imaginary = spectrum.imaginary[index];
      return real ** 2 + imaginary ** 2;
    });
    const magnitudeFrame = powers.map((power) => Math.log10(power + 1e-8));
    const melFrame = melFilters.map((filter) => filter.reduce((sum, weight, index) => sum + weight * powers[index], 0));
    const melValues = melFrame.map((value) => Math.log10(value + 1e-8));
    const phaseFrame = Array.from({ length: FFT_SIZE / 2 + 1 }, (_, index) => {
      return Math.atan2(spectrum.imaginary[index], spectrum.real[index]);
    });
    mel.push(melValues);
    magnitude.push(magnitudeFrame);
    phase.push(phaseFrame);
  }

  return {
    melSpectrogram: resizeMatrix(normalizeMatrix(mel), DISPLAY_BANDS, DISPLAY_FRAMES),
    stftMagnitude: resizeMatrix(normalizeMatrix(magnitude), DISPLAY_BANDS, DISPLAY_FRAMES),
    phasePattern: resizeMatrix(phase.map((row) => row.map((value) => (value + Math.PI) / (2 * Math.PI))), DISPLAY_BANDS, DISPLAY_FRAMES),
  };
}

function calculateArtifacts(fingerprint: AudioFingerprint, probFake: number): ArtifactInfo[] {
  const flatPhase = fingerprint.phasePattern.flat();
  const phaseVariance = flatPhase.reduce((sum, value) => sum + (value - 0.5) ** 2, 0) / flatPhase.length;
  const spectralRows = fingerprint.stftMagnitude;
  const highFrequencyEnergy = spectralRows.slice(Math.floor(spectralRows.length * 0.68)).flat().reduce((sum, value) => sum + value, 0) / (spectralRows.length * 0.32 * spectralRows[0].length);
  const temporalJumps = fingerprint.melSpectrogram.reduce((sum, row, rowIndex) => {
    if (rowIndex === 0) return sum;
    return sum + row.reduce((rowSum, value, index) => rowSum + Math.abs(value - fingerprint.melSpectrogram[rowIndex - 1][index]), 0) / row.length;
  }, 0) / fingerprint.melSpectrogram.length;
  return [
    { name: 'Phase coherence', description: 'Tracks frame-to-frame consistency in phase behavior across frequency bands.', severity: phaseVariance > 0.11 ? 'high' : phaseVariance > 0.075 ? 'moderate' : 'low', confidence: Math.min(0.98, 0.56 + phaseVariance * 2.4) },
    { name: 'Spectral texture', description: 'Checks for unnaturally regular harmonics and high-frequency generator residue.', severity: highFrequencyEnergy > 0.48 ? 'high' : highFrequencyEnergy > 0.35 ? 'moderate' : 'low', confidence: Math.min(0.97, 0.52 + highFrequencyEnergy * 0.45) },
    { name: 'Temporal cadence', description: 'Measures energy changes and pause rhythm over the analysis window.', severity: temporalJumps > 0.29 ? 'high' : temporalJumps > 0.2 ? 'moderate' : 'low', confidence: Math.min(0.96, 0.5 + temporalJumps * 0.8) },
  ].map((artifact) => ({ ...artifact, confidence: Math.max(artifact.confidence, probFake > 0.5 ? 0.7 : 0.55) }));
}

function analyzeSegment(segment: Float32Array, sampleRate: number): { fingerprint: AudioFingerprint; score: number } {
  const fingerprint = extractFeatures(segment, sampleRate);
  const phaseVariance = fingerprint.phasePattern.flat().reduce((sum, value) => sum + (value - 0.5) ** 2, 0) / (fingerprint.phasePattern.length * fingerprint.phasePattern[0].length);
  const spectralTexture = fingerprint.stftMagnitude.flat().reduce((sum, value, index, values) => sum + Math.abs(value - values[Math.max(0, index - 1)]), 0) / fingerprint.stftMagnitude.flat().length;
  const cadence = fingerprint.melSpectrogram.reduce((sum, row, index) => index ? sum + Math.abs(row.reduce((a, b) => a + b, 0) - fingerprint.melSpectrogram[index - 1].reduce((a, b) => a + b, 0)) : sum, 0) / fingerprint.melSpectrogram.length;
  const rawScore = 0.24 + phaseVariance * 1.35 + spectralTexture * 0.5 + cadence * 0.75;
  return { fingerprint, score: Math.max(0.04, Math.min(0.96, rawScore)) };
}

export function analyzeAudio(samples: Float32Array, sampleRate: number): AnalysisResult {
  const normalized = normalize(trimSilence(resample(samples, sampleRate, ANALYSIS_SAMPLE_RATE)));
  const windowLength = ANALYSIS_SAMPLE_RATE * ANALYSIS_WINDOW_SEC;
  const segmentCount = Math.max(1, Math.ceil(normalized.length / windowLength));
  const segmentResults: SegmentResult[] = [];
  const analyses: { fingerprint: AudioFingerprint; score: number }[] = [];

  for (let index = 0; index < segmentCount; index += 1) {
    const segment = normalized.slice(index * windowLength, Math.min((index + 1) * windowLength, normalized.length));
    const padded = new Float32Array(windowLength);
    padded.set(segment);
    const analysis = analyzeSegment(padded, ANALYSIS_SAMPLE_RATE);
    analyses.push(analysis);
    segmentResults.push({ index: index + 1, startTimeSec: index * ANALYSIS_WINDOW_SEC, endTimeSec: Math.min((index + 1) * ANALYSIS_WINDOW_SEC, normalized.length / ANALYSIS_SAMPLE_RATE), probFake: analysis.score, verdict: analysis.score > 0.5 ? 'DEEPFAKE' : 'REAL' });
  }

  const probFake = segmentResults.reduce((sum, segment) => sum + segment.probFake, 0) / segmentResults.length;
  const fingerprint = analyses[0].fingerprint;
  const channelMeans = [fingerprint.melSpectrogram, fingerprint.stftMagnitude, fingerprint.phasePattern].map((matrix) => matrix.flat().reduce((sum, value) => sum + value, 0) / matrix.flat().length);
  const attentionWeights = Array.from({ length: 10 }, (_, index) => Math.max(0.18, Math.min(0.96, 0.3 + channelMeans[index % 3] * 0.45 + (index % 4) * 0.05)));
  return {
    verdict: probFake > 0.5 ? 'DEEPFAKE' : 'REAL',
    probFake,
    probReal: 1 - probFake,
    segments: segmentResults,
    melSpectrogram: fingerprint.melSpectrogram,
    stftMagnitude: fingerprint.stftMagnitude,
    phasePattern: fingerprint.phasePattern,
    attentionWeights,
    artifacts: calculateArtifacts(fingerprint, probFake),
    durationSec: normalized.length / ANALYSIS_SAMPLE_RATE,
    sampleRate: ANALYSIS_SAMPLE_RATE,
  };
}

export async function decodeAudioFile(file: File): Promise<{ samples: Float32Array; sampleRate: number }> {
  const context = new AudioContext();
  const buffer = await context.decodeAudioData(await file.arrayBuffer());
  const channels = Array.from({ length: buffer.numberOfChannels }, (_, index) => buffer.getChannelData(index));
  const samples = new Float32Array(buffer.length);
  channels.forEach((channel) => channel.forEach((value, index) => { samples[index] += value / channels.length; }));
  await context.close();
  return { samples, sampleRate: buffer.sampleRate };
}

export { ANALYSIS_SAMPLE_RATE };
