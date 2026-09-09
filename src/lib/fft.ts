export function fft(real: Float32Array): { real: Float32Array; imaginary: Float32Array } {
  const n = real.length;
  if (n === 1) return { real: new Float32Array([real[0]]), imaginary: new Float32Array([0]) };
  if ((n & (n - 1)) !== 0) throw new Error('FFT input length must be a power of two');

  const even = new Float32Array(n / 2);
  const odd = new Float32Array(n / 2);
  for (let i = 0; i < n / 2; i += 1) {
    even[i] = real[i * 2];
    odd[i] = real[i * 2 + 1];
  }

  const evenFft = fft(even);
  const oddFft = fft(odd);
  const outputReal = new Float32Array(n);
  const outputImaginary = new Float32Array(n);

  for (let k = 0; k < n / 2; k += 1) {
    const angle = (-2 * Math.PI * k) / n;
    const cosine = Math.cos(angle);
    const sine = Math.sin(angle);
    const oddReal = oddFft.real[k] * cosine - oddFft.imaginary[k] * sine;
    const oddImaginary = oddFft.real[k] * sine + oddFft.imaginary[k] * cosine;
    outputReal[k] = evenFft.real[k] + oddReal;
    outputImaginary[k] = evenFft.imaginary[k] + oddImaginary;
    outputReal[k + n / 2] = evenFft.real[k] - oddReal;
    outputImaginary[k + n / 2] = evenFft.imaginary[k] - oddImaginary;
  }

  return { real: outputReal, imaginary: outputImaginary };
}

export function hannWindow(length: number): Float32Array {
  const window = new Float32Array(length);
  for (let i = 0; i < length; i += 1) {
    window[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (length - 1)));
  }
  return window;
}

export function frequencyToMel(frequency: number): number {
  return 2595 * Math.log10(1 + frequency / 700);
}

export function melToFrequency(mel: number): number {
  return 700 * (10 ** (mel / 2595) - 1);
}

export function createMelFilterBank(
  sampleRate: number,
  fftSize: number,
  filterCount: number,
  minFrequency = 40,
  maxFrequency = sampleRate / 2,
): Float32Array[] {
  const minMel = frequencyToMel(minFrequency);
  const maxMel = frequencyToMel(maxFrequency);
  const melPoints = Array.from({ length: filterCount + 2 }, (_, index) => {
    const mel = minMel + ((maxMel - minMel) * index) / (filterCount + 1);
    return melToFrequency(mel);
  });
  const bins = melPoints.map((frequency) => Math.floor(((fftSize + 1) * frequency) / sampleRate));
  const filters: Float32Array[] = [];

  for (let filterIndex = 1; filterIndex <= filterCount; filterIndex += 1) {
    const filter = new Float32Array(fftSize / 2 + 1);
    const left = bins[filterIndex - 1];
    const center = Math.max(bins[filterIndex], left + 1);
    const right = Math.max(bins[filterIndex + 1], center + 1);
    for (let bin = left; bin < center && bin < filter.length; bin += 1) {
      filter[bin] = (bin - left) / (center - left);
    }
    for (let bin = center; bin < right && bin < filter.length; bin += 1) {
      filter[bin] = (right - bin) / (right - center);
    }
    filters.push(filter);
  }
  return filters;
}
