"use client";
import { useState } from "react";

export default function Home() {
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setImagePreview(URL.createObjectURL(file));
      setResult(null); // Reset hasil lama
    }
  };

  const handlePredict = async () => {
    const fileInput = document.getElementById("fileUpload") as HTMLInputElement;
    const file = fileInput?.files?.[0];
    if (!file) return alert("Pilih gambar dulu!");

    setLoading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("/api/detect", {
        method: "POST",
        body: formData,
      });
      
      const data = await res.json();
      setResult(data);
    } catch (err) {
      console.error(err);
      alert("Gagal mendeteksi warna");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gray-100 p-5">
      <div className="bg-white p-8 rounded-xl shadow-lg max-w-md w-full text-center">
        <h1 className="text-2xl font-bold mb-5 text-gray-800">Cek Warna Dominan</h1>
        
        <div className="w-full h-64 bg-gray-200 rounded-lg mb-5 overflow-hidden flex items-center justify-center border border-gray-300">
          {imagePreview ? (
            <img src={imagePreview} alt="Preview" className="w-full h-full object-contain" />
          ) : (
            <span className="text-gray-500">Preview Gambar</span>
          )}
        </div>

        <input 
          id="fileUpload" 
          type="file" 
          accept="image/*" 
          onChange={handleFileChange}
          className="block w-full text-sm text-gray-500 mb-4 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-gray-50 file:text-gray-700 hover:file:bg-gray-100"
        />

        <button
          onClick={handlePredict}
          disabled={loading || !imagePreview}
          className="w-full bg-gray-600 hover:bg-gray-700 text-white font-bold py-3 rounded-lg transition disabled:bg-gray-400"
        >
          {loading ? "Sedang Menganalisis..." : "Deteksi Warna"}
        </button>

        {result && (
          <div className="mt-6 p-4 bg-gray-50 border border-gray-200 rounded-lg">
            <p className="text-sm text-gray-600">Warna Terdeteksi:</p>
            <h2 className="text-3xl font-extrabold text-gray-700 uppercase mt-1">
              {result.label}
            </h2>
            <p className="text-xs text-gray-500 mt-2">
              Kemiripan: {(result.confidence * 100).toFixed(1)}%
            </p>
          </div>
        )}
      </div>
    </div>
  );
}