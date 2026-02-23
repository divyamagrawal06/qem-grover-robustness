"use client";

import { useEffect, useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

interface ApiStatus {
  status: string;
  project: string;
}

export default function Home() {
  const [apiStatus, setApiStatus] = useState<ApiStatus | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${API_BASE}/`)
      .then((res) => res.json())
      .then((data: ApiStatus) => setApiStatus(data))
      .catch(() => setError("Backend unreachable — start it with uvicorn"));
  }, []);

  return (
    <div className="flex min-h-screen flex-col items-center justify-center bg-gradient-to-br from-gray-950 via-gray-900 to-indigo-950 text-white font-sans">
      {/* ── Hero ─────────────────────────────────────── */}
      <main className="flex flex-col items-center gap-8 px-6 text-center">
        <div className="relative">
          <div className="absolute -inset-4 rounded-full bg-indigo-500/20 blur-3xl" />
          <h1 className="relative text-5xl font-bold tracking-tight sm:text-6xl">
            <span className="bg-gradient-to-r from-indigo-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
              QEM-Grover
            </span>{" "}
            Robustness
          </h1>
        </div>

        <p className="max-w-xl text-lg leading-relaxed text-gray-400">
          A systematic study of how{" "}
          <span className="text-white font-medium">
            Grover&apos;s Algorithm
          </span>{" "}
          breaks down under depolarizing noise, readout errors, and decoherence
          — plus a deep-learning QEM pipeline to fight back.
        </p>

        {/* ── Status pill ─────────────────────────────── */}
        <div className="mt-4 flex items-center gap-3 rounded-full border border-white/10 bg-white/5 px-5 py-2.5 text-sm backdrop-blur">
          <span
            className={`h-2.5 w-2.5 rounded-full ${
              apiStatus
                ? "bg-emerald-400 shadow-[0_0_6px_theme(colors.emerald.400)]"
                : error
                  ? "bg-red-400 shadow-[0_0_6px_theme(colors.red.400)]"
                  : "animate-pulse bg-amber-400"
            }`}
          />
          <span className="text-gray-300">
            {apiStatus
              ? `API connected — ${apiStatus.project}`
              : error
                ? error
                : "Connecting to backend…"}
          </span>
        </div>

        {/* ── Quick links ─────────────────────────────── */}
        <div className="mt-6 flex flex-wrap justify-center gap-4">
          <a
            href="/api-docs"
            className="rounded-lg border border-indigo-500/30 bg-indigo-500/10 px-5 py-2.5 text-sm font-medium text-indigo-300 transition hover:bg-indigo-500/20"
          >
            API Docs ↗
          </a>
          <a
            href="http://localhost:8000/docs"
            target="_blank"
            rel="noopener noreferrer"
            className="rounded-lg border border-white/10 bg-white/5 px-5 py-2.5 text-sm font-medium text-gray-300 transition hover:bg-white/10"
          >
            Swagger UI ↗
          </a>
        </div>
      </main>

      {/* ── Footer ────────────────────────────────────── */}
      <footer className="absolute bottom-6 text-xs text-gray-600">
        QEM-Grover Robustness · Research Dashboard
      </footer>
    </div>
  );
}
