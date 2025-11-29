import type { NextConfig } from "next";

const nextConfig: NextConfig = {

  serverExternalPackages: ['onnxruntime-node', 'sharp'],
  
  // experimental: {
  //   serverComponentsExternalPackages: ['onnxruntime-node', 'sharp'],
  // },
};

export default nextConfig;