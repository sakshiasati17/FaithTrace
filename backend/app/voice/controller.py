"""
Voice Controller — orchestrates the full voice interaction loop:
  VAD → Whisper STT → Intent Parser → FaithTrace API → TTS response
"""

import json

import requests

from .intent_parser import IntentParser, ParsedIntent
from .stt import SpeechToText, VoiceActivityDetector
from .tts import TextToSpeech


class VoiceController:
    def __init__(
        self,
        api_base_url: str = "http://localhost:8000/api/v1",
        whisper_model: str = "base",
        device: str = "cuda",
    ):
        print("Initialising FaithTrace Voice Controller…")
        self.api_base_url = api_base_url
        self.stt = SpeechToText(model_size=whisper_model, device=device)
        self.vad = VoiceActivityDetector()
        self.tts = TextToSpeech()
        self.parser = IntentParser()
        print("Voice Controller ready.")

    def run(self):
        self.tts.speak("FaithTrace voice interface ready. How can I help?")

        while True:
            try:
                audio = self.vad.listen()
                if audio is None:
                    continue

                transcription = self.stt.transcribe_audio(audio)
                print(f"You said: '{transcription.text}'")

                if any(w in transcription.text.lower() for w in ["exit", "quit", "stop", "goodbye", "bye"]):
                    self.tts.speak("Goodbye!")
                    break

                intent = self.parser.parse(transcription.text)
                print(f"Intent: {intent.action} | Params: {intent.parameters}")

                response_text = self._handle_intent(intent)
                print(f"Response: {response_text}")
                self.tts.speak(response_text)

            except KeyboardInterrupt:
                print("\nVoice controller stopped.")
                break
            except Exception as e:
                print(f"Error: {e}")
                self.tts.speak("Sorry, I encountered an error. Please try again.")

    def _handle_intent(self, intent: ParsedIntent) -> str:
        if intent.action == "unknown":
            return f"I didn't understand: {intent.raw_text}. Say help to hear available commands."

        if intent.action == "help":
            return self.parser.help_text()

        try:
            url = f"{self.api_base_url}{intent.api_endpoint}"
            if intent.api_method == "GET":
                resp = requests.get(url, params=intent.parameters, timeout=30)
            elif intent.api_method == "POST":
                resp = requests.post(url, json=intent.parameters, timeout=60)
            else:
                return "Unsupported operation."

            if resp.status_code == 200:
                return self._format_response(intent.action, resp.json())
            return f"API returned error {resp.status_code}."

        except requests.exceptions.ConnectionError:
            return "Cannot connect to FaithTrace API. Is the server running?"
        except requests.exceptions.Timeout:
            return "The request timed out. The operation may still be running."

    def _format_response(self, action: str, data: dict) -> str:
        if action == "run_experiment":
            return f"Experiment started. Task ID: {data.get('task_id', 'unknown')}."

        if action == "diagnose_latest":
            return self.tts.format_diagnostic_response(data)

        if action == "compare_experiments":
            a = data.get("experiment_a", {})
            b = data.get("experiment_b", {})
            winner = "A" if a.get("avg_score", 0) > b.get("avg_score", 0) else "B"
            return (
                f"Experiment {winner} performed better. "
                f"A scored {a.get('avg_score', 0):.2f}, "
                f"B scored {b.get('avg_score', 0):.2f}."
            )

        if action == "get_failures":
            count = len(data.get("failures", []))
            top = data.get("most_common_failure", "unknown")
            return f"Found {count} failed queries. Most common failure: {top.replace('_', ' ')}."

        if action == "get_recommendations":
            recs = data.get("recommendations", [])
            if not recs:
                return "No recommendations at this time. The pipeline looks good."
            top = recs[0]
            return f"Top recommendation: {top.get('description', '')}. Impact: {top.get('expected_impact', '')}."

        if action == "gpu_profile":
            return (
                f"Detected {data.get('name', 'unknown GPU')} with "
                f"{data.get('vram_available_gb', 0)} GB available. "
                f"Recommended precision: {data.get('recommended_precision', 'fp32').upper()}."
            )

        if action == "run_benchmark":
            return "Benchmark started. This may take a few minutes."

        if action == "get_benchmark":
            results = data.get("results", [])
            if results:
                fastest = min(results, key=lambda r: r.get("p50_ms", 999))
                return (
                    f"Best config: {fastest['label']} — "
                    f"{fastest['p50_ms']} ms median, "
                    f"{fastest.get('throughput_qps', 0)} queries per second."
                )
            return "No benchmark results yet."

        if action == "check_status":
            return (
                f"{data.get('active_jobs', 0)} jobs running. "
                f"{data.get('completed_today', 0)} experiments completed today."
            )

        return f"Done. {json.dumps(data)[:200]}"


def start_voice_interface():
    controller = VoiceController()
    controller.run()


if __name__ == "__main__":
    start_voice_interface()
