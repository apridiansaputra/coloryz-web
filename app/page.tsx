"use client";
import { useState, useEffect, useRef } from "react";
import * as ort from "onnxruntime-web";
import { processImage } from "./utils/imageHelper";

// --- KONFIGURASI ---
const LABELS = ["Beige", "Black", "Blue", "Gray", "Green", "Pattern", "Red", "White"];

// Fungsi Softmax (Matematika Persentase)
function softmax(logits: Float32Array): number[] {
  let maxLogit = -Infinity;
  for (let i = 0; i < logits.length; i++) if (logits[i] > maxLogit) maxLogit = logits[i];
  const exps = [];
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    const e = Math.exp(logits[i] - maxLogit);
    exps.push(e);
    sum += e;
  }
  return exps.map(e => e / sum);
}

export default function Home() {
  // State Logika
  const [session, setSession] = useState<ort.InferenceSession | null>(null);
  const [modelStatus, setModelStatus] = useState("Memuat AI...");
  const [isModelReady, setIsModelReady] = useState(false);

  // State UI (Pindah Halaman)
  const [view, setView] = useState<"HOME" | "RESULT">("HOME"); // Mengatur tampilan
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [resultData, setResultData] = useState<{ label: string; score: string } | null>(null);
  const [loading, setLoading] = useState(false);

  const imgRef = useRef<HTMLImageElement>(null);

  // 1. Load Model saat awal buka
  useEffect(() => {
    async function initModel() {
      try {
        // Ganti URL ini dengan file onnx kamu di folder public
        const sess = await ort.InferenceSession.create("/model/color_resnet50.onnx", {
          executionProviders: ["wasm"],
        });
        setSession(sess);
        setIsModelReady(true);
        setModelStatus("AI Siap");
      } catch (e) {
        console.error(e);
        setModelStatus("Gagal Memuat Model");
      }
    }
    initModel();
  }, []);

  // 2. Handle Upload Gambar
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const url = URL.createObjectURL(file);
      setImagePreview(url);
      // Langsung proses setelah pilih gambar (User Experience lebih cepat)
      handlePredict(url); 
    }
  };

  // 3. Proses Prediksi
  const handlePredict = async (url: string) => {
    setLoading(true);
    // Kita butuh elemen gambar tersembunyi untuk dibaca canvas
    const img = new Image();
    img.src = url;
    img.onload = async () => {
      try {
        if (!session) return;
        
        // Proses AI
        const inputTensor = await processImage(img);
        const feeds = { input: inputTensor };
        const results = await session.run(feeds);
        const outputKey = session.outputNames[0];
        const outputData = results[outputKey].data as Float32Array;
        
        // Hitung Hasil
        const probabilities = softmax(outputData);
        let maxIdx = 0;
        let maxVal = probabilities[0];
        for(let i = 1; i < probabilities.length; i++){
          if(probabilities[i] > maxVal) {
            maxVal = probabilities[i];
            maxIdx = i;
          }
        }

        // Set Hasil & Pindah Halaman
        setResultData({
          label: LABELS[maxIdx],
          score: (maxVal * 100).toFixed(0) + "%"
        });
        setView("RESULT"); // Pindah ke halaman hasil

      } catch (e) {
        alert("Gagal memproses gambar.");
        console.error(e);
      } finally {
        setLoading(false);
      }
    };
  };

  // 4. Tombol Kembali
  const handleReset = () => {
    setImagePreview(null);
    setResultData(null);
    setView("HOME");
  };

  // --- HALAMAN 1: HOME ---
  if (view === "HOME") {
    return (
      <div className="min-h-screen bg-white flex flex-col relative overflow-hidden font-poppins">
        {/* Hiasan Background Blob */}
        <div className="absolute top-0 right-0 w-64 h-64 bg-blue-100 rounded-full mix-blend-multiply filter blur-3xl opacity-70 animate-blob"></div>
        <div className="absolute -bottom-8 -left-8 w-72 h-72 bg-purple-100 rounded-full mix-blend-multiply filter blur-3xl opacity-70 animate-blob animation-delay-2000"></div>

        <main className="flex-1 flex flex-col items-center justify-center p-6 text-center z-10">
          {/* Logo / Header */}
          <div className="mb-8 animate-slide-up">
            <h1 className="text-5xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-purple-600 mb-2">
              Coloryz
            </h1>
            <p className="text-gray-500 text-sm max-w-xs mx-auto">
              Solusi visual cerdas untuk membantu identifikasi warna secara instan.
            </p>
          </div>

          {/* Status Indicator */}
          {!isModelReady && (
            <div className="mb-8 px-4 py-2 bg-yellow-50 text-yellow-700 rounded-full text-xs font-medium animate-pulse">
              ⏳ {modelStatus} (Sedang mengunduh aset...)
            </div>
          )}

          {/* Upload Card / Button */}
          <div className="w-full max-w-sm animate-fade-in delay-100">
            <label className={`
              group flex flex-col items-center justify-center w-full h-64 
              border-2 border-dashed border-blue-200 rounded-3xl 
              bg-blue-50/50 cursor-pointer 
              transition-all duration-300
              hover:bg-blue-50 hover:border-blue-400 hover:shadow-lg hover:-translate-y-1
              ${!isModelReady ? 'opacity-50 pointer-events-none' : ''}
            `}>
              <div className="flex flex-col items-center justify-center pt-5 pb-6">
                <div className="w-16 h-16 mb-4 bg-white rounded-full flex items-center justify-center shadow-md group-hover:scale-110 transition-transform">
                  <svg className="w-8 h-8 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 4v16m8-8H4"></path></svg>
                </div>
                <p className="mb-2 text-lg font-semibold text-gray-700">Upload Gambar</p>
                <p className="text-xs text-gray-500">Tap di sini untuk mulai analisis</p>
              </div>
              <input type="file" className="hidden" accept="image/*" onChange={handleFileChange} />
            </label>
          </div>

          {/* Footer Text */}
          <div className="mt-12 text-xs text-gray-400">
            Created by <span className="font-semibold text-gray-600">Spectova Team</span>
          </div>
        </main>

        {/* Loading Overlay saat proses */}
        {loading && (
          <div className="absolute inset-0 bg-white/80 backdrop-blur-sm z-50 flex flex-col items-center justify-center">
            <div className="w-12 h-12 border-4 border-blue-200 border-t-blue-600 rounded-full animate-spin mb-4"></div>
            <p className="text-blue-600 font-semibold animate-pulse">Menganalisis Warna...</p>
          </div>
        )}
      </div>
    );
  }

  // --- HALAMAN 2: RESULT ---
  return (
    <div className="min-h-screen bg-gray-50 flex flex-col font-poppins">
      {/* Header Kecil */}
      <div className="bg-white px-6 py-4 shadow-sm flex items-center justify-between z-20">
        <h2 className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-purple-600">Coloryz</h2>
        <button onClick={handleReset} className="text-sm text-gray-500 hover:text-gray-800">
          ✕ Tutup
        </button>
      </div>

      <main className="flex-1 flex flex-col p-6 animate-fade-in">
        
        {/* Card Gambar */}
        <div className="flex-1 bg-white rounded-3xl shadow-xl overflow-hidden relative mb-6 border border-gray-100 flex items-center justify-center bg-checkered">
          {imagePreview && (
            <img 
              src={imagePreview} 
              alt="Uploaded" 
              className="w-full h-full object-contain max-h-[50vh]" 
            />
          )}
        </div>

        {/* Card Hasil Deteksi */}
        <div className="bg-white rounded-3xl p-6 shadow-lg border border-gray-100 animate-slide-up">
          <p className="text-gray-400 text-xs font-semibold uppercase tracking-wider mb-2">
            Warna Dominan Terdeteksi
          </p>
          
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-4xl font-bold text-gray-800 mb-1">
                {resultData?.label}
              </h1>
              <p className="text-sm text-green-600 font-medium bg-green-50 inline-block px-2 py-1 rounded-md">
                Confidence: {resultData?.score}
              </p>
            </div>
            
            {/* Indikator Warna Visual (Opsional) */}
            <div className="w-16 h-16 rounded-2xl shadow-inner border border-gray-100" 
                 style={{ backgroundColor: resultData?.label.toLowerCase() === 'pattern' ? 'gray' : resultData?.label.toLowerCase() }}>
            </div>
          </div>

          <div className="mt-6">
            <button 
              onClick={handleReset}
              className="w-full py-4 rounded-xl bg-gradient-to-r from-blue-600 to-purple-600 text-white font-bold shadow-lg shadow-blue-200 hover:shadow-xl hover:scale-[1.02] transition-all duration-300"
            >
              Coba Gambar Lain
            </button>
          </div>
        </div>

        <div className="mt-6 text-center text-xs text-gray-400">
          Solusi visual by Spectova
        </div>
      </main>
    </div>
  );
}