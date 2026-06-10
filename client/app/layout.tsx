import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "HouseMusic.ai",
  description:
    "Ask about house music — its history, its clubs, its DJs. Answers grounded in real sources.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        {/*
          Fonts are loaded via <link> rather than next/font so the build never
          hard-fails when Google Fonts is unreachable (e.g. offline / CI). If
          the fetch fails at runtime, the CSS stack in globals.css falls back to
          system faces cleanly. Self-host these for production if you prefer.
        */}
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link
          rel="preconnect"
          href="https://fonts.gstatic.com"
          crossOrigin=""
        />
        <link
          href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="font-body antialiased">{children}</body>
    </html>
  );
}