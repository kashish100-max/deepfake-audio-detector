export type Verdict = 'REAL' | 'DEEPFAKE';

export type AnalysisResult = {
  verdict: Verdict;
  probFake: number;
  probReal: number;
  segments: SegmentResult[];
  melSpectrogram: number[][];
  stftMagnitude: number[][];
  phasePattern: number[][];
  attentionWeights: number[];
  artifacts: ArtifactInfo[];
  durationSec: number;
  sampleRate: number;
};

export type SegmentResult = {
  index: number;
  startTimeSec: number;
  endTimeSec: number;
  probFake: number;
  verdict: Verdict;
};

export type ArtifactInfo = {
  name: string;
  description: string;
  severity: 'low' | 'moderate' | 'high';
  confidence: number;
};

export type AudioFingerprint = {
  melSpectrogram: number[][];
  stftMagnitude: number[][];
  phasePattern: number[][];
};
