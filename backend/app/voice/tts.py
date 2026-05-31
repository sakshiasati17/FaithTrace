"""
Text-to-Speech for FaithTrace voice responses.

Development: Coqui TTS (open source, local GPU).
Production upgrade path: NVIDIA Riva TTS (streaming synthesis,
  lower latency, higher quality, gRPC API — swap is straightforward).
"""


class TextToSpeech:
    def __init__(self, model_name: str = "tts_models/en/ljspeech/tacotron2-DDC"):
        try:
            from TTS.api import TTS
            print(f"Loading Coqui TTS: {model_name}")
            self.tts = TTS(model_name=model_name, progress_bar=False)
            self.engine = "coqui"
            self.sample_rate = 22_050
            print("Coqui TTS ready.")
        except ImportError:
            import pyttsx3
            print("Coqui TTS not installed, using pyttsx3 fallback.")
            self.tts = pyttsx3.init()
            self.tts.setProperty("rate", 160)
            self.engine = "pyttsx3"
            self.sample_rate = 22_050

    def speak(self, text: str):
        if self.engine == "coqui":
            import numpy as np
            import sounddevice as sd
            wav = self.tts.tts(text=text)
            sd.play(np.array(wav, dtype=np.float32), samplerate=self.sample_rate)
            sd.wait()
        else:
            self.tts.say(text)
            self.tts.runAndWait()

    def speak_to_file(self, text: str, output_path: str):
        if self.engine == "coqui":
            self.tts.tts_to_file(text=text, file_path=output_path)
        print(f"Audio saved: {output_path}")

    def format_diagnostic_response(self, diagnostic_result: dict) -> str:
        """Convert raw diagnostic dict to natural spoken English."""
        failure_type = diagnostic_result.get("failure_type", "unknown")
        confidence = diagnostic_result.get("confidence", 0)
        confidence_pct = int(confidence * 100)

        descriptions = {
            "no_failure": "The answer looks correct and well-grounded in the retrieved context.",
            "retrieval_miss": (
                "The retrieval system failed to find the relevant document. "
                "The answer may be based on irrelevant context."
            ),
            "context_insufficient": (
                "The right document was retrieved but the specific chunk "
                "was missing key information needed for a complete answer."
            ),
            "hallucination": (
                "The generated answer contained claims not supported by "
                "the retrieved context. The model fabricated information."
            ),
            "prompt_weakness": (
                "The model misunderstood the task due to poor prompt design. "
                "Consider revising the system prompt."
            ),
            "ranking_failure": (
                "Relevant information was retrieved but ranked too low. "
                "The model focused on irrelevant chunks instead."
            ),
        }

        description = descriptions.get(failure_type, "Unknown failure type detected.")
        return (
            f"The analysis found a {failure_type.replace('_', ' ')} issue "
            f"with {confidence_pct} percent confidence. {description}"
        )
