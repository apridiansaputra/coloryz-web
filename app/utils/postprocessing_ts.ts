
// Utility untuk postprocessing output ONNX di Node.js / Next.js.
// Berisi softmax dan fungsi untuk memilih label terbaik.

/**
 * Softmax yang numerically stable.
 */
export function softmax(logits: number[]): number[] {
  const maxLogit = Math.max(...logits);
  const exps = logits.map((v) => Math.exp(v - maxLogit));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

/**
 * Ambil label terbaik dari logits dan daftar label.
 * logits: output mentah model (tanpa softmax)
 * labels: array string label (index sama dengan urutan training)
 */
export function predictLabel(logits: number[], labels: string[]) {
  const probs = softmax(logits);

  let bestIdx = 0;
  let bestProb = probs[0];

  for (let i = 1; i < probs.length; i++) {
    if (probs[i] > bestProb) {
      bestProb = probs[i];
      bestIdx = i;
    }
  }

  const label = labels[bestIdx];

  return {
    label,               // nama kelas
    confidence: bestProb, // probabilitas tertinggi
    index: bestIdx,       // index kelas
  };
}
