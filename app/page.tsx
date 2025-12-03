"use client";
import { useState, useRef, useEffect } from "react";
import * as ort from "onnxruntime-web";
import { processImage } from "./utils/imageHelper";
import { Krona_One } from "next/font/google";

const kronaOne = Krona_One({ 
  weight: "400", 
  subsets: ["latin"],
  variable: "--font-krona",
});

const LABELS = ["Beige", "Black", "Blue", "Gray", "Green", "Pattern", "Red", "White"];

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
  const [view, setView] = useState<"HOME" | "RESULT">("HOME");
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [resultData, setResultData] = useState<{ label: string; score: string } | null>(null);
  const [loading, setLoading] = useState(false);
  const [session, setSession] = useState<ort.InferenceSession | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    async function initModel() {
      try {
        const sess = await ort.InferenceSession.create("/model/color_resnet50.onnx", {
          executionProviders: ["wasm"],
        });
        setSession(sess);
      } catch (e) {
        console.error("Gagal load model", e);
      }
    }
    initModel();
  }, []);


  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const url = URL.createObjectURL(file);
      setImagePreview(url);
      setSelectedFile(file);
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile || !imagePreview) {
      alert("Silakan unggah gambar terlebih dahulu sebelum mendeteksi.");
      return;
    }
    
    setLoading(true);
    const img = new Image();
    img.src = imagePreview;
    
    img.onload = async () => {
      try {
        if (!session) {
          setTimeout(() => {
            setResultData({ label: "Loading...", score: "0%" });
            setView("RESULT");
            setLoading(false);
          }, 1000);
          return;
        }

        const inputTensor = await processImage(img);
        const feeds = { input: inputTensor };
        const results = await session.run(feeds);
        const outputData = results[session.outputNames[0]].data as Float32Array;
        
        const probs = softmax(outputData);
        let maxIdx = 0, maxVal = probs[0];
        for(let i=1; i<probs.length; i++) {
          if(probs[i] > maxVal) { maxVal = probs[i]; maxIdx = i; }
        }

        setResultData({
          label: LABELS[maxIdx],
          score: (maxVal * 100).toFixed(0) + "%"
        });
        setView("RESULT");
      } catch (e) {
        console.error(e);
        alert("Gagal memproses gambar.");
      } finally {
        setLoading(false);
      }
    };
  };

  const handleReset = () => {
    setImagePreview(null);
    setSelectedFile(null);
    setResultData(null);
    setView("HOME");
  };

  if (view === "HOME") {
    return (
      <div className={`min-h-screen relative overflow-hidden font-sans ${kronaOne.variable}`} style={{
        background: 'linear-gradient(to bottom, #05123D 0%, #05123D 60%, #009BFF 100%)'
      }}>
        <main className="flex flex-col items-center justify-between min-h-screen p-6 text-center relative z-10 max-w-md mx-auto w-full">
          
          <div className="w-full pt-8 flex justify-start">
            <h1 className="text-xl mb-2 tracking-wide" style={{
              fontFamily: 'var(--font-krona)', 
              color: '#7B61FF' 
            }}>
              ColoryZ
            </h1>
          </div>

          <div className="flex-1 flex flex-col justify-center w-full">
            
            <p className="text-white text-md mb-10 mt-10 opacity-90">
              Mengidentifikasi warna di dalam gambar menggunakan teknologi AI dengan cepat dan akurat, serta memberikan informasi warna yang jelas terutama bagi pengguna dengan kondisi defisiensi penglihatan warna.
            </p>

            <div className="flex flex-col gap-4 mb-10 items-center w-full">
              <div className="flex gap-3 justify-center w-full">
                <div className="bg-white/5 backdrop-blur-md rounded-xl px-3 py-2.5 flex items-center gap-2 border flex-1 justify-center min-w-0" 
                    style={{ borderColor: '#7B61FF' }}>
                  <svg className="w-4 h-4 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20" style={{ color: '#7B61FF' }}>
                    <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd"/>
                  </svg>
                  <span className="text-white text-[10px] sm:text-xs font-medium text-left leading-tight">Kejelasan Warna</span>
                </div>

                <div className="bg-white/5 backdrop-blur-md rounded-xl px-3 py-2.5 flex items-center gap-2 border flex-1 justify-center min-w-0" 
                    style={{ borderColor: '#05D588' }}>
                  <svg className="w-4 h-4 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20" style={{ color: '#05D588' }}>
                    <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd"/>
                  </svg>
                  <span className="text-white text-[10px] sm:text-xs font-medium text-left leading-tight">Warna Dominan</span>
                </div>
              </div>

              
              <div className="flex justify-center w-full">
                <div className="bg-white/5 backdrop-blur-md rounded-xl px-4 py-2.5 flex items-center gap-2 border min-w-[50%]" 
                    style={{ borderColor: '#43FFFC' }}>
                  <svg className="w-4 h-4 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20" style={{ color: '#43FFFC' }}>
                    <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd"/>
                  </svg>
                  <span className="text-white text-xs font-medium text-left">Identifikasi Instan AI</span>
                </div>
              </div>
            </div>

            <div 
              className="w-full mb-6 cursor-pointer group"
              onClick={() => fileInputRef.current?.click()}
            >
              <div className={`
                bg-white/5 backdrop-blur-sm border-2 border-dashed rounded-3xl p-8 
                flex flex-col items-center justify-center transition-all duration-300 
                ${selectedFile ? 'border-[#05D588] bg-white/10' : 'border-white/20 group-hover:bg-white/10 group-hover:border-white/40'}
              `}>
                {selectedFile && imagePreview ? (
                   <div className="w-full flex flex-col items-center animate-fade-in">
                      <img src={imagePreview} alt="Selected" className="w-32 h-32 object-contain rounded-lg mb-2 shadow-lg" />
                      <p className="text-[#05D588] text-sm font-bold">Gambar Terpilih!</p>
                      <p className="text-white/50 text-xs">Ketuk untuk ganti</p>
                   </div>
                ) : (
                   <>
                    <div className="w-16 h-16 bg-white/10 rounded-full flex items-center justify-center mb-4 shadow-lg">
                      <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"/>
                      </svg>
                    </div>
                    <p className="text-white text-base font-medium group-hover:scale-105 transition-transform">Klik Untuk Unggah Gambar / Kamera</p>
                    <p className="text-white/50 text-xs mt-2">Mendukung JPG, PNG</p>
                   </>
                )}
              </div>
              <input 
                ref={fileInputRef}
                type="file" 
                className="hidden" 
                accept="image/*" 
                onChange={handleFileChange} 
              />
            </div>

            
            <div className="w-full">
              <button 
                className="w-full rounded-full py-4 px-6 text-white font-bold text-lg shadow-lg shadow-indigo-900/50 transition-all duration-300 hover:scale-[1.02] active:scale-95" 
                style={{ backgroundColor: '#7B61FF' }}
                onClick={handleAnalyze}
              >
                Deteksi Warna
              </button>
            </div>

          </div>

          <div className="w-full pb-2 pt-8">
            <div className="flex flex-col items-start text-white/60 text-xs gap-1">
               <p className="font-semibold tracking-wide">Created by Spectova</p>
               <div className="flex justify-between w-full items-end">
                  <p className="text-white/40 text-[10px]">Apridian, Lathief, Keti, Shintya</p>
                  <div className="flex items-center gap-1 text-white/40">
                    <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20"><path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-12a1 1 0 10-2 0v4a1 1 0 00.293.707l2.828 2.829a1 1 0 101.415-1.415L11 9.586V6z" clipRule="evenodd"/></svg>
                    <span>2025</span>
                  </div>
               </div>
            </div>
          </div>

          {loading && (
            <div className="absolute inset-0 z-50 flex flex-col items-center justify-center bg-black/80 backdrop-blur-md">
              <div className="w-12 h-12 border-4 border-white/20 border-t-[#7B61FF] rounded-full animate-spin mb-4"></div>
              <p className="text-white font-semibold animate-pulse">Menganalisis Warna...</p>
            </div>
          )}
        </main>
      </div>
    );
  }

  
  return (
    <div className={`min-h-screen relative overflow-hidden font-sans ${kronaOne.variable}`} style={{
      background: 'linear-gradient(to bottom, #05123D 0%, #05123D 60%, #009BFF 100%)'
    }}>
      <div className="w-full max-w-md mx-auto h-full flex flex-col min-h-screen">
        <div className="px-6 py-6 flex items-center justify-between relative z-20">
          <h2 className="text-xl" style={{ fontFamily: 'var(--font-krona)', color: '#7B61FF' }}>ColoryZ</h2>
          <div className="flex items-center gap-1">
            <button onClick={handleReset} className="text-white hover:bg-white/10 p-2 rounded-full transition">
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 19l-7-7 7-7"/>
              </svg>
            </button>
            <button onClick={handleReset} className="text-white/80 text-sm hover:text-white transition">
            Beranda</button>
          </div>
        </div>

        <main className="flex-1 flex flex-col p-6 pb-8 h-full">
          <div className="bg-black/20 backdrop-blur-md rounded-[2rem] overflow-hidden mb-8 border border-white/10 shadow-2xl flex items-center justify-center relative min-h-[300px]">
            {imagePreview ? (
              <img 
                src={imagePreview} 
                alt="Uploaded" 
                className="w-full h-full object-contain max-h-[400px]" 
              />
            ) : (
              <div className="text-white/50">Tidak ada gambar</div>
            )}
          </div>

          <div className="flex items-start gap-3 mb-8 px-2 bg-white/5 p-4 rounded-xl border border-white/5">
            <svg className="w-5 h-5 text-[#43FFFC] flex-shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <p className="text-white/80 text-xs leading-relaxed">
              Hasil analisis ini berbasis pada warna dominan objek di dalam gambar yang terdeteksi oleh AI.
            </p>
          </div>

          
          <div className="mb-8 text-center animate-slide-up">
             <h3 className="text-2xl font-bold text-white tracking-wide">
                Hasil Deteksi Warna : {resultData?.label} ({resultData?.score})
             </h3>
          </div>

          
          <button 
            onClick={handleReset}
            className="w-full rounded-full py-4 px-6 text-white font-bold transition-all duration-300 hover:scale-[1.02] flex items-center justify-center gap-2 shadow-lg"
            style={{ backgroundColor: '#7B61FF' }}
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"/>
            </svg>
            Deteksi Gambar Lain
          </button>

          <div className="mt-8 text-center">
              <p className="text-white/30 text-[10px]">Spectova AI © 2025</p>
          </div>
        </main>
      </div>
    </div>
  );
}