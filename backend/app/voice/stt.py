"""
Speech-to-Text using OpenAI Whisper (local GPU inference).

Model size guide:
  tiny / base  — fast, good for short clear commands
  small / medium — best balance for FaithTrace commands
  large — highest accuracy, 10x slower, overkill for short commands

WHY LOCAL WHISPER over the API:
- No per-call cost.
- Lower latency for short utterances on a local GPU.
- Swap path to NVIDIA Riva for production streaming STT is straightforward
  (same intent-parser interface, just replace the transcription call).
"""

import queue
from dataclasses import dataclass

import numpy as np


@dataclass
class TranscriptionResult:
    text: str
    language: str
    confidence: float
    duration_seconds: float


class SpeechToText:
    def __init__(self, model_size: str = "base", device: str = "cuda"):
        import whisper
        print(f"Loading Whisper {model_size}…")
        self.model = whisper.load_model(model_size, device=device)
        self.sample_rate = 16_000
        self.device = device
        print("Whisper ready.")

    def transcribe_audio(self, audio_array: np.ndarray) -> TranscriptionResult:
        """Transcribe a mono float32 16 kHz numpy array."""
        if audio_array.dtype != np.float32:
            audio_array = audio_array.astype(np.float32)
        if np.max(np.abs(audio_array)) > 1.0:
            audio_array = audio_array / np.max(np.abs(audio_array))

        result = self.model.transcribe(
            audio_array,
            language="en",
            fp16=(self.device == "cuda"),
            task="transcribe",
        )

        segs = result.get("segments", [])
        avg_logprob = np.mean([s.get("avg_logprob", -1.0) for s in segs]) if segs else -1.0
        confidence = float(np.clip(np.exp(avg_logprob), 0, 1))

        return TranscriptionResult(
            text=result["text"].strip(),
            language=result.get("language", "en"),
            confidence=round(confidence, 3),
            duration_seconds=round(len(audio_array) / self.sample_rate, 2),
        )

    def transcribe_file(self, audio_path: str) -> TranscriptionResult:
        result = self.model.transcribe(audio_path, language="en")
        return TranscriptionResult(
            text=result["text"].strip(),
            language=result.get("language", "en"),
            confidence=0.0,
            duration_seconds=0.0,
        )


class VoiceActivityDetector:
    """
    Energy-based VAD — detects speech start/end from microphone input.

    Production upgrade path: replace with Silero VAD or WebRTC VAD for
    noise-robust detection without manual threshold tuning.
    """

    def __init__(
        self,
        sample_rate: int = 16_000,
        energy_threshold: float = 0.02,
        silence_duration: float = 1.5,
        min_speech_duration: float = 0.5,
        max_speech_duration: float = 30.0,
    ):
        self.sample_rate = sample_rate
        self.energy_threshold = energy_threshold
        self.silence_duration = silence_duration
        self.min_speech_duration = min_speech_duration
        self.max_speech_duration = max_speech_duration
        self.chunk_duration = 0.1
        self.chunk_size = int(sample_rate * self.chunk_duration)

    def listen(self):
        import sounddevice as sd

        audio_buffer = []
        is_speaking = False
        silence_chunks = 0
        max_silence = int(self.silence_duration / self.chunk_duration)
        max_total = int(self.max_speech_duration / self.chunk_duration)
        total_chunks = 0
        audio_q: queue.Queue = queue.Queue()

        def callback(indata, frames, time_info, status):
            audio_q.put(indata.copy())

        print("Listening… (speak now)")
        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.chunk_size,
            callback=callback,
        ):
            while total_chunks < max_total:
                try:
                    chunk = audio_q.get(timeout=5.0)
                except queue.Empty:
                    if not is_speaking:
                        continue
                    break

                energy = float(np.sqrt(np.mean(chunk ** 2)))
                if energy > self.energy_threshold:
                    if not is_speaking:
                        print("Speech detected.")
                        is_speaking = True
                    silence_chunks = 0
                    audio_buffer.append(chunk)
                elif is_speaking:
                    silence_chunks += 1
                    audio_buffer.append(chunk)
                    if silence_chunks >= max_silence:
                        print("Speech ended.")
                        break
                total_chunks += 1

        if not audio_buffer:
            return None

        audio = np.concatenate(audio_buffer).flatten()
        duration = len(audio) / self.sample_rate
        if duration < self.min_speech_duration:
            print(f"Too short ({duration:.1f}s), ignoring.")
            return None

        print(f"Captured {duration:.1f}s")
        return audio
