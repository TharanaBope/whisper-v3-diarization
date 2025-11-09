import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Audio Transcription & Diarization",
  description: "AI-powered audio transcription with speaker identification",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>
        {children}
      </body>
    </html>
  );
}
