"use client";
import { useState, useEffect, useRef } from "react";
import * as ort from "onnxruntime-web";
import { processImage } from "./utils/imageHelper";

const LABELS = ["beige", "black", "blue", "gray", "green", "pattern", "red", "white"];

// Fungsi Matematika: Mengubah Skor Mentah -> Persentase (0-100%)
function softmax(logits: Float32Array): number[] {
  let maxLogit = -Infinity;
  // 1. Cari nilai terbesar (untuk kestabilan numerik)
  for (let i = 0; i < logits.length; i++) {
    if (logits[i] > maxLogit) maxLogit = logits[i];
  }
  
  // 2. Hitung Eksponensial
  const exps = [];
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    const e = Math.exp(logits[i] - maxLogit);
    exps.push(e);
    sum += e;
  }
  
  // 3. Bagi dengan total (agar jumlahnya jadi 1.0 atau 100%)
  return exps.map(e => e / sum);
}

export default function Home() {
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [result, setResult] = useState<string>("");
  const [loading, setLoading] = useState(false);
  
  // Status Loading Model
  const [modelStatus, setModelStatus] = useState("Memulai...");
  const [downloadProgress, setDownloadProgress] = useState(false);
  
  const [session, setSession] = useState<ort.InferenceSession | null>(null);
  const imgRef = useRef<HTMLImageElement>(null);

  useEffect(() => {
    async function initModel() {
      try {
        setModelStatus("Sedang mendownload Model AI (100MB)... Harap tunggu.");
        setDownloadProgress(true);

        const sess = await ort.InferenceSession.create("/models/color_resnet50.onnx", {
          executionProviders: ["wasm"], 
        });
        
        setSession(sess);
        setModelStatus("✅ Siap Digunakan!");
        setDownloadProgress(false);
      } catch (e) {
        console.error("Error load model:", e);
        setModelStatus("❌ Gagal memuat model. Cek koneksi internet.");
        setDownloadProgress(false);
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
      // 1. Proses Gambar
      const inputTensor = await processImage(imgRef.current);

      // 2. Jalankan AI
      const feeds = { input: inputTensor };
      const results = await session.run(feeds);
      
      // 3. Ambil Output Mentah
      const outputKey = session.outputNames[0];
      const outputData = results[outputKey].data as Float32Array;

      // 4. HITUNG SOFTMAX (Perbaikan disini)
      const probabilities = softmax(outputData);

      // 5. Cari nilai tertinggi dari probabilitas
      let maxIdx = 0;
      let maxVal = probabilities[0];
      for(let i = 1; i < probabilities.length; i++){
          if(probabilities[i] > maxVal) {
              maxVal = probabilities[i];
              maxIdx = i;
          }
      }

      // 6. Tampilkan Hasil
      // maxVal sekarang adalah 0.0 sampai 1.0 (contoh 0.95)
      const percentage = (maxVal * 100).toFixed(1); 
      setResult(`${LABELS[maxIdx]} (${percentage}%)`);

    } catch (e) {
      console.error(e);
      alert("Gagal mendeteksi.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gray-50 p-6 font-sans">
      <div className="bg-white p-8 rounded-2xl shadow-xl w-full max-w-md text-center border border-gray-100">
        <h1 className="text-3xl font-bold mb-2 text-blue-600 tracking-tight">Coloryz</h1>
        
        {/* Status Bar */}
        <div className={`text-sm mb-6 py-2 px-4 rounded-full inline-block ${downloadProgress ? 'bg-yellow-100 text-yellow-700 animate-pulse' : 'bg-gray-100 text-gray-600'}`}>
           {modelStatus}
        </div>
        
        {/* Preview Area */}
        <div className="relative w-full h-72 bg-gray-100 rounded-xl mb-6 flex items-center justify-center overflow-hidden border border-gray-200 group">
            {imagePreview ? (
                <img 
                    ref={imgRef} 
                    src={imagePreview} 
                    alt="Preview" 
                    className="w-full h-full object-contain"
                />
            ) : (
                <div className="text-gray-400 flex flex-col items-center">
                    <svg className="w-12 h-12 mb-2 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" /></svg>
                    <span>Pilih gambar untuk mulai</span>
                </div>
            )}
        </div>

        {/* Input File */}
        <label className="block mb-4 cursor-pointer">
            <span className="sr-only">Pilih gambar</span>
            <input type="file" onChange={handleFileChange} accept="image/*" className="block w-full text-sm text-gray-500
              file:mr-4 file:py-2.5 file:px-6
              file:rounded-full file:border-0
              file:text-sm file:font-semibold
              file:bg-blue-50 file:text-blue-700
              hover:file:bg-blue-100 transition-colors
            "/>
        </label>
        
        <button 
            onClick={handlePredict} 
            disabled={loading || !session || !imagePreview}
            className="w-full bg-blue-600 hover:bg-blue-700 text-white py-3.5 rounded-xl font-bold transition-all disabled:bg-gray-300 disabled:cursor-not-allowed shadow-md hover:shadow-lg active:scale-95"
        >
            {loading ? (
                <span className="flex items-center justify-center">
                    <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
                    Menganalisis...
                </span>
            ) : "Deteksi Warna"}
        </button>

        {/* Result Card */}
        {result && (
            <div className="mt-8 p-6 bg-green-50 border border-green-200 rounded-xl animate-[fadeIn_0.5s_ease-out]">
                <p className="text-xs font-bold text-green-600 uppercase tracking-widest mb-1">Hasil Deteksi</p>
                <h2 className="text-4xl font-extrabold text-gray-800 capitalize mb-1">{result.split(' ')[0]}</h2>
                <div className="inline-block px-3 py-1 bg-green-200 text-green-800 text-xs font-bold rounded-full">
                    Akurasi {result.split(' ')[1]}
                </div>
            </div>
        )}
      </div>
    </div>
  );
}