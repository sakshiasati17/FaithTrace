"use client";

import { useState, useRef, useCallback } from "react";
import { Mic, MicOff, Loader2, Volume2 } from "lucide-react";
import { voiceApi } from "@/lib/api";
import { clsx } from "clsx";
import type { VoiceCommandResult } from "@/types";

type RecordingState = "idle" | "recording" | "processing";

export function VoicePanel() {
  const [state, setState] = useState<RecordingState>("idle");
  const [result, setResult] = useState<VoiceCommandResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  const startRecording = useCallback(async () => {
    setError(null);
    setResult(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
      chunksRef.current = [];

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      recorder.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop());
        setState("processing");
        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        try {
          const res = await voiceApi.command(blob);
          setResult(res);
        } catch (err: any) {
          setError(err.message ?? "Command failed");
        } finally {
          setState("idle");
        }
      };

      mediaRecorderRef.current = recorder;
      recorder.start();
      setState("recording");
    } catch (err: any) {
      setError(err.message ?? "Microphone access denied");
    }
  }, []);

  const stopRecording = useCallback(() => {
    mediaRecorderRef.current?.stop();
  }, []);

  return (
    <div className="rounded-xl bg-zinc-900 border border-zinc-800 p-5">
      <div className="flex items-center gap-2 mb-4">
        <Mic className="w-4 h-4 text-amber-400" />
        <h2 className="text-sm font-semibold text-zinc-200">Voice Command</h2>
        <span className="text-[10px] px-1.5 py-0.5 rounded bg-amber-900/30 text-amber-400 border border-amber-700/40 font-medium ml-auto">
          Whisper STT
        </span>
      </div>

      {/* Record button */}
      <div className="flex items-center gap-4 mb-4">
        {state === "recording" ? (
          <button
            onClick={stopRecording}
            className="inline-flex items-center gap-2 px-4 py-2 text-sm font-semibold text-white rounded-lg transition-all animate-pulse"
            style={{ background: "linear-gradient(135deg,#dc2626,#b91c1c)" }}
          >
            <MicOff className="w-3.5 h-3.5" />
            Stop Recording
          </button>
        ) : state === "processing" ? (
          <button disabled className="inline-flex items-center gap-2 px-4 py-2 text-sm font-semibold text-zinc-500 rounded-lg bg-zinc-800 border border-zinc-700 cursor-not-allowed">
            <Loader2 className="w-3.5 h-3.5 animate-spin" />
            Processing…
          </button>
        ) : (
          <button
            onClick={startRecording}
            className="inline-flex items-center gap-2 px-4 py-2 text-sm font-semibold text-white rounded-lg transition-all"
            style={{ background: "linear-gradient(135deg,#d97706,#b45309)" }}
          >
            <Mic className="w-3.5 h-3.5" />
            Hold to Record
          </button>
        )}

        {state === "recording" && (
          <span className="flex items-center gap-1.5 text-xs text-red-400">
            <span className="w-1.5 h-1.5 rounded-full bg-red-500 animate-pulse" />
            Recording…
          </span>
        )}
      </div>

      {/* Error */}
      {error && (
        <div className="mb-3 px-3 py-2 rounded-lg bg-red-900/20 border border-red-700/40 text-xs text-red-400">
          {error}
        </div>
      )}

      {/* Result */}
      {result && (
        <div className="space-y-2.5">
          <div className="p-3 rounded-lg bg-zinc-800/60 border border-zinc-700/50">
            <p className="text-[10px] text-zinc-500 uppercase tracking-wider mb-1">You said</p>
            <p className="text-sm text-zinc-200 italic">"{result.transcription}"</p>
            <p className="text-[11px] text-zinc-600 mt-0.5">
              Confidence: {(result.confidence * 100).toFixed(0)}% · Language: {result.language}
            </p>
          </div>

          <div className="p-3 rounded-lg bg-zinc-800/60 border border-zinc-700/50">
            <p className="text-[10px] text-zinc-500 uppercase tracking-wider mb-1">Intent</p>
            <span className="text-xs font-mono px-2 py-0.5 rounded bg-violet-900/30 text-violet-300 border border-violet-700/40">
              {result.intent}
            </span>
            {result.api_endpoint && (
              <span className="ml-2 text-[11px] font-mono text-zinc-500">
                {result.api_method} {result.api_endpoint}
              </span>
            )}
          </div>

          {(result as any).spoken_response && (
            <div className="p-3 rounded-lg bg-emerald-900/10 border border-emerald-700/30">
              <div className="flex items-center gap-1.5 mb-1">
                <Volume2 className="w-3 h-3 text-emerald-400" />
                <p className="text-[10px] text-emerald-500 uppercase tracking-wider">Response</p>
              </div>
              <p className="text-sm text-emerald-300">{(result as any).spoken_response}</p>
            </div>
          )}
        </div>
      )}

      {!result && !error && state === "idle" && (
        <p className="text-xs text-zinc-600">
          Say: <span className="font-mono text-zinc-500">"what went wrong"</span>,{" "}
          <span className="font-mono text-zinc-500">"show failed queries"</span>,{" "}
          <span className="font-mono text-zinc-500">"suggest improvements"</span>…
        </p>
      )}
    </div>
  );
}
