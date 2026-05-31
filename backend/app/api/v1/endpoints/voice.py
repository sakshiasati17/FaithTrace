"""
Voice interface REST endpoints.

POST /voice/command  — accept audio file, return transcription + parsed intent
POST /voice/speak    — convert text to WAV and return audio stream
GET  /voice/health   — Whisper model status
"""

import io

import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

router = APIRouter()

# Lazy-loaded singletons so the model only loads when the endpoint is first called
_stt = None
_tts = None
_parser = None


def _get_stt():
    global _stt
    if _stt is None:
        from app.voice.stt import SpeechToText
        _stt = SpeechToText(model_size="base", device="cpu")
    return _stt


def _get_tts():
    global _tts
    if _tts is None:
        from app.voice.tts import TextToSpeech
        _tts = TextToSpeech()
    return _tts


def _get_parser():
    global _parser
    if _parser is None:
        from app.voice.intent_parser import IntentParser
        _parser = IntentParser()
    return _parser


@router.post("/command")
async def voice_command(audio: UploadFile = File(...)):
    """
    Accept an audio file (WAV/MP3/OGG), transcribe it with Whisper,
    parse the intent, and return structured JSON.

    The frontend can record browser audio via MediaRecorder API and POST it here,
    enabling voice interaction without a local microphone loop.
    """
    try:
        import soundfile as sf
    except ImportError:
        raise HTTPException(status_code=500, detail="soundfile not installed")

    audio_bytes = await audio.read()
    try:
        audio_array, sample_rate = sf.read(io.BytesIO(audio_bytes))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse audio: {e}")

    # Resample to 16 kHz if needed (Whisper requirement)
    if sample_rate != 16_000:
        try:
            import librosa
            audio_array = librosa.resample(audio_array.astype(np.float32), orig_sr=sample_rate, target_sr=16_000)
        except ImportError:
            raise HTTPException(status_code=500, detail="librosa not installed; audio must be 16 kHz")

    transcription = _get_stt().transcribe_audio(audio_array.astype(np.float32))
    intent = _get_parser().parse(transcription.text)

    return {
        "transcription": transcription.text,
        "language": transcription.language,
        "confidence": transcription.confidence,
        "intent": intent.action,
        "parameters": intent.parameters,
        "api_endpoint": intent.api_endpoint,
        "api_method": intent.api_method,
    }


@router.post("/speak")
async def text_to_speech(text: str):
    """Convert text to a WAV audio stream via Coqui TTS."""
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        _get_tts().speak_to_file(text, tmp_path)
        with open(tmp_path, "rb") as f:
            audio_bytes = f.read()
    finally:
        os.unlink(tmp_path)

    return StreamingResponse(
        io.BytesIO(audio_bytes),
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=response.wav"},
    )


@router.get("/health")
async def voice_health():
    """Return whether voice models are loaded."""
    return {
        "stt_loaded": _stt is not None,
        "tts_loaded": _tts is not None,
        "whisper_model": "base",
        "tts_engine": _tts.engine if _tts else "not loaded",
    }
