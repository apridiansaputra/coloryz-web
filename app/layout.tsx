import type { Metadata } from "next";
import { Poppins } from "next/font/google"; 
import "./globals.css";

const poppins = Poppins({ 
  subsets: ["latin"],
  weight: ["300", "400", "600", "700"], 
  variable: "--font-poppins",
});

export const metadata: Metadata = {
  title: "Coloryz by Spectova",
  description: "AI Color Detection for Color Blindness Aid",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${poppins.variable} font-sans antialiased bg-gray-50`}>
        {children}
      </body>
    </html>
  );
}