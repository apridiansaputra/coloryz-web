export const runtime = 'nodejs';

import { NextRequest, NextResponse } from "next/server";
import * as ort from "onnxruntime-node";
import path from "path";
import fs from "fs/promises";
import { preprocessImageBuffer } from "../../utils/preprocessing_ts";
import { predictLabel } from "../../utils/postprocessing_ts";       


let session: ort.InferenceSession | null = null;
let labels: string[] | null = null;

async function loadModel() {
  if (!session || !labels) {
    
    const modelPath = path.join(process.cwd(), "public", "model", "color_resnet50.onnx");
    const labelsPath = path.join(process.cwd(), "public", "model", "labels.json");

    // Load ONNX Session
    session = await ort.InferenceSession.create(modelPath);
    
    // Load Labels
    const labelsData = await fs.readFile(labelsPath, "utf-8");
    labels = JSON.parse(labelsData);
    console.log("Model & Labels loaded server-side!");
  }
  return { session, labels };
}

export async function POST(req: NextRequest) {
  try {
    // 1. Ambil file gambar dari request
    const formData = await req.formData();
    const file = formData.get("file") as File;

    if (!file) {
      return NextResponse.json({ error: "No file uploaded" }, { status: 400 });
    }

    // 2. Ubah File jadi Buffer (biar bisa dimakan sama 'sharp')
    const arrayBuffer = await file.arrayBuffer();
    const buffer = Buffer.from(arrayBuffer);

    // 3. Preprocessing (Pake script temanmu)
    // Script temanmu return Float32Array, kita bungkus jadi Tensor
    const float32Data = await preprocessImageBuffer(buffer);
    const inputTensor = new ort.Tensor("float32", float32Data, [1, 3, 224, 224]);

    // 4. Load Model & Run Inference
    const { session, labels } = await loadModel();
    if (!session || !labels) throw new Error("Model failed to load");

    // 'input' adalah nama layer input model (biasanya 'input' atau 'input.1'). 
    // Kalau error, cek nama input modelnya pake Netron.app, tapi biasanya ResNet standar itu 'input'.
    // Berdasarkan snippet ONNX kamu: nama inputnya sepertinya "input" (dari snippet dynamic_axes)
    const feeds = { input: inputTensor }; 
    const results = await session.run(feeds);
    
    // Ambil output (biasanya nama outputnya 'output' atau 'fc')
    // Kita ambil value pertama dari results object
    const outputKey = session.outputNames[0];
    const outputData = results[outputKey].data as Float32Array;

    // 5. Postprocessing (Pake script temanmu)
    // Convert Float32Array ke normal array number[]
    const outputArray = Array.from(outputData);
    const result = predictLabel(outputArray, labels);

    return NextResponse.json(result);

  } catch (error) {
    console.error("Prediction Error:", error);
    return NextResponse.json({ error: "Internal Server Error" }, { status: 500 });
  }
}