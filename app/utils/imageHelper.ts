import * as ort from "onnxruntime-web";

// Standar Normalisasi ImageNet (Wajib untuk ResNet)
const MEAN = [0.485, 0.456, 0.406];
const STD  = [0.229, 0.224, 0.225];

export async function processImage(imageElement: HTMLImageElement): Promise<ort.Tensor> {
  const size = 224;

  // 1. Buat Canvas virtual di memori browser
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");
  
  if (!ctx) throw new Error("Browser tidak mendukung Canvas 2D");

  // 2. Gambar foto ke canvas (Otomatis resize ke 224x224)
  ctx.drawImage(imageElement, 0, 0, size, size);

  // 3. Ambil data Pixel mentah (RGBA)
  const imageData = ctx.getImageData(0, 0, size, size);
  const { data } = imageData; // Array panjang

  // 4. Konversi ke Tensor (Format CHW - Channel, Height, Width)
  // Ini menggantikan fungsi 'sharp' dan 'preprocessing.ts' yang lama
  const float32Data = new Float32Array(size * size * 3);

  for (let i = 0; i < size * size; i++) {
    const r = data[i * 4] / 255;     // Red (0-1)
    const g = data[i * 4 + 1] / 255; // Green (0-1)
    const b = data[i * 4 + 2] / 255; // Blue (0-1)

    // Masukkan ke array Float32 dengan urutan CHW (Penting untuk PyTorch/ONNX)
    // Channel Red
    float32Data[i] = (r - MEAN[0]) / STD[0];
    // Channel Green
    float32Data[i + size * size] = (g - MEAN[1]) / STD[1];
    // Channel Blue
    float32Data[i + size * size * 2] = (b - MEAN[2]) / STD[2];
  }

  // 5. Return Tensor ONNX [1, 3, 224, 224]
  return new ort.Tensor("float32", float32Data, [1, 3, 224, 224]);
}