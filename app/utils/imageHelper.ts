import * as ort from "onnxruntime-web";

const MEAN = [0.485, 0.456, 0.406];
const STD  = [0.229, 0.224, 0.225];

export async function processImage(imageElement: HTMLImageElement): Promise<ort.Tensor> {
  const size = 224;

  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");
  
  if (!ctx) throw new Error("Browser tidak mendukung Canvas 2D");

  ctx.drawImage(imageElement, 0, 0, size, size);

  const imageData = ctx.getImageData(0, 0, size, size);
  const { data } = imageData; 

  const float32Data = new Float32Array(size * size * 3);

  for (let i = 0; i < size * size; i++) {
    const r = data[i * 4] / 255;     
    const g = data[i * 4 + 1] / 255; 
    const b = data[i * 4 + 2] / 255; 

    float32Data[i] = (r - MEAN[0]) / STD[0];
    float32Data[i + size * size] = (g - MEAN[1]) / STD[1];
    float32Data[i + size * size * 2] = (b - MEAN[2]) / STD[2];
  }

  return new ort.Tensor("float32", float32Data, [1, 3, 224, 224]);
}