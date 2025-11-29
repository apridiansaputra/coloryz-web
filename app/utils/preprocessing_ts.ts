
// Utility untuk preprocessing gambar di sisi Node.js / Next.js API.
// Catatan:
// - membutuhkan package: sharp
//   npm install sharp
//
// Cara pakai (contoh di API Next.js):
//   import { preprocessImageBuffer } from "./preprocessing_ts";
//   const inputTensorData = await preprocessImageBuffer(buffer);
//   const inputTensor = new ort.Tensor("float32", inputTensorData, [1, 3, 224, 224]);

import sharp from "sharp";

const MEAN = [0.485, 0.456, 0.406];
const STD  = [0.229, 0.224, 0.225];

/**
 * Preprocess buffer gambar (mis. hasil upload) menjadi Float32Array
 * dengan shape implicit (1, 3, 224, 224) sesuai kebutuhan ONNX.
 */
export async function preprocessImageBuffer(buffer: Buffer): Promise<Float32Array> {
  const size = 224;

  // Resize dan ambil raw RGB (tanpa alpha, tanpa header)
  const img = await sharp(buffer)
    .resize(size, size)
    .removeAlpha()
    .raw()
    .toBuffer(); // panjang = size*size*3

  const floatData = new Float32Array(size * size * 3);

  // Normalisasi per channel (HWC)
  for (let i = 0; i < size * size; i++) {
    const r = img[i * 3] / 255;
    const g = img[i * 3 + 1] / 255;
    const b = img[i * 3 + 2] / 255;

    floatData[i * 3]     = (r - MEAN[0]) / STD[0];
    floatData[i * 3 + 1] = (g - MEAN[1]) / STD[1];
    floatData[i * 3 + 2] = (b - MEAN[2]) / STD[2];
  }

  // HWC -> CHW + batch: (1, 3, H, W) flattened
  const chwData = new Float32Array(1 * 3 * size * size);
  let dstIndex = 0;

  for (let c = 0; c < 3; c++) {
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const srcIndex = (y * size + x) * 3 + c; // HWC
        chwData[dstIndex++] = floatData[srcIndex];
      }
    }
  }

  return chwData;
}
