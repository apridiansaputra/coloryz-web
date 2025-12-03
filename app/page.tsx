"use client";
import { useState, useEffect, useRef } from "react";
import * as ort from "onnxruntime-web";
import { processImage } from "./utils/imageHelper";

// Daftar Label Manual (Supaya tidak perlu baca file JSON yang bikin ribet)
const LABELS = ["beige", "black", "blue", "gray", "green", "pattern", "red", "white"];

export default function Home() {
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [result, setResult] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState("Memuat Model AI...");
  const [session, setSession] = useState<ort.InferenceSession | null>(null);
  
  // Ref untuk elemen gambar (agar bisa dibaca helper)
  const imgRef = useRef<HTMLImageElement>(null);

  // 1. Load Model saat website dibuka
  useEffect(() => {
    async function initModel() {
      try {
        // Mengambil model dari folder public/models/
        // Pastikan file .onnx kamu ada di folder: public/models/color_resnet50.onnx
        const sess = await ort.InferenceSession.create("/models/color_resnet50.onnx", {
          executionProviders: ["wasm"], // Menggunakan CPU Browser
        });
        setSession(sess);
        setStatus("Siap Deteksi!");
      } catch (e) {
        console.error("Gagal load model:", e);
        setStatus("Gagal memuat model. Cek console.");
      }
    }
    initModel();
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setImagePreview(URL.createObjectURL(file));
      setResult("");
    }
  };

  const handlePredict = async () => {
    if (!session || !imgRef.current) return;
    setLoading(true);

    try {
      // 1. Proses Gambar pakai Helper (Client Side)
      const inputTensor = await processImage(imgRef.current);

      // 2. Jalankan AI
      const feeds = { input: inputTensor }; // 'input' adalah nama layer standar ResNet
      const results = await session.run(feeds);
      
      // 3. Ambil Output
      const outputKey = session.outputNames[0];
      const outputData = results[outputKey].data as Float32Array;

      // 4. Cari nilai tertinggi (Argmax)
      let maxIdx = 0;
      let maxVal = outputData[0];
      for(let i=1; i<outputData.length; i++){
          if(outputData[i] > maxVal) {
              maxVal = outputData[i];
              maxIdx = i;
          }
      }

      // 5. Tampilkan Hasil
      setResult(`${LABELS[maxIdx]} (${(maxVal * 100).toFixed(0)}%)`);

    } catch (e) {
      console.error(e);
      alert("Gagal mendeteksi. Cek console browser (F12).");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gray-50 p-6">
      <div className="bg-white p-8 rounded-2xl shadow-xl w-full max-w-md text-center">
        <h1 className="text-2xl font-bold mb-2 text-blue-600">Coloryz</h1>
        <p className="text-sm text-gray-500 mb-6">{status}</p>
        
        {/* Gambar Preview */}
        <div className="w-full h-64 bg-gray-100 rounded-lg mb-6 flex items-center justify-center overflow-hidden border">
            {imagePreview ? (
                <img ref={imgRef} src={imagePreview} alt="Preview" className="w-full h-full object-contain" />
            ) : (
                <span className="text-gray-400">Pilih gambar...</span>
            )}
        </div>

        <input type="file" onChange={handleFileChange} className="mb-4 block w-full text-sm text-gray-500" />
        
        <button 
            onClick={handlePredict} 
            disabled={loading || !session || !imagePreview}
            className="w-full bg-blue-600 text-white py-3 rounded-lg font-bold disabled:bg-gray-300"
        >
            {loading ? "Menganalisis..." : "Deteksi Warna"}
        </button>

        {result && (
            <div className="mt-6 p-4 bg-green-50 border border-green-200 rounded-xl">
                <h2 className="text-3xl font-extrabold text-green-700 uppercase">{result}</h2>
            </div>
        )}
      </div>
    </div>
  );
}