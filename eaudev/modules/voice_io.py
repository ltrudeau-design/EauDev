"""
VoiceIO — voice input/output mode for EauDev.

Integrates Silero VAD, faster-whisper ASR, and Piper TTS as an optional
I/O mode. Activated via /voice or --voice. Uses EauDev's already-running
llama.cpp inference server — no second model needed.

Architecture:
    Microphone → VAD (Silero) → ASR (faster-whisper) → EauDev agent loop
    EauDev response → TTS (Piper) → Speaker

TTS streaming:
    One Piper process per response. Full text written to stdin at once.
    PCM chunks streamed directly to sounddevice.RawOutputStream as they
    arrive — audio starts playing while Piper is still synthesizing the
    tail end of a long response. No gaps, no buffering delay.

Dependencies (optional — imported lazily):
    torch           — Silero VAD
    faster_whisper  — ASR
    sounddevice     — audio I/O
    numpy           — audio buffers
    piper CLI       — TTS (must be on PATH)

All dependencies are optional. If missing, VoiceIO reports the gap and
refuses to activate rather than crashing EauDev.
"""
from __future__ import annotations

import shutil
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

# ── Configuration ──────────────────────────────────────────────────────────────

@dataclass
class VoiceIOConfig:
    # ASR
    whisper_model: str = "base"       # tiny / base / small / medium
    whisper_language: str = "en"
    whisper_compute_type: str = "int8" # int8 = fastest on CPU/M1

    # VAD (original Specialist settings for better voice detection)
    vad_threshold: float = 0.5         # Higher = less sensitive, fewer false triggers
    vad_min_silence_ms: int = 300      # Shorter = faster response
    vad_min_speech_ms: int = 250       # minimum speech duration
    vad_padding_ms: int = 250          # silence padding added to utterance edges

    # TTS
    piper_model: str = ""  # path to .onnx piper model — set in ~/.eaudev/config.yml under voice_io.piper_model
    piper_cmd: str = "piper"
    piper_speaker_id: Optional[int] = None
    piper_length_scale: float = 1.0
    piper_noise_scale: float = 0.667
    piper_noise_w: float = 0.8
    piper_sample_rate: int = 16000     # Default for most Piper models (check .onnx.json)

    # Audio device
    sample_rate: int = 16000
    channels: int = 1
    blocksize: int = 512               # must match Silero's expected 512 samples

    # Behaviour
    print_transcript: bool = True      # echo [you said: ...] to terminal
    speak_responses: bool = True       # TTS output
    print_responses: bool = True       # print response even when speaking


# ── Dependency check ───────────────────────────────────────────────────────────

def check_dependencies() -> list[str]:
    """Return list of missing dependency descriptions. Empty = all present."""
    missing: list[str] = []
    try:
        import torch  # noqa: F401
    except ImportError:
        missing.append("torch  (pip install torch)")
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        missing.append("faster_whisper  (pip install faster-whisper)")
    try:
        import sounddevice  # noqa: F401
    except ImportError:
        missing.append("sounddevice  (pip install sounddevice)")
    try:
        import numpy  # noqa: F401
    except ImportError:
        missing.append("numpy  (pip install numpy)")
    if not shutil.which("piper"):
        missing.append("piper CLI  (pip install piper-tts)")
    return missing


# ── VAD engine ─────────────────────────────────────────────────────────────────

class _VADEngine:
    """Silero VAD — segments microphone stream into speech utterances."""

    # Silero processes exactly 512 samples at 16kHz = 32ms per frame
    _FRAME_MS: int = 32

    def __init__(self, threshold: float, min_silence_ms: int, min_speech_ms: int,
                 padding_ms: int) -> None:
        import torch
        self.threshold = threshold
        self.max_silence_frames = max(1, min_silence_ms // self._FRAME_MS)
        self.padding_samples = int(padding_ms / 1000 * 16000)

        self.model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
        )
        self.model.eval()

    def create_stream_callback(
        self,
        on_speech_start: Callable[[], None],
        on_speech_end: Callable,
    ) -> Callable:
        import numpy as np
        import torch

        buffer: list = []
        is_speaking = False
        silence_frames = 0

        def callback(indata, frames, time, status) -> None:
            nonlocal buffer, is_speaking, silence_frames

            audio = indata.squeeze()

            # Silero requires exactly 512 samples at 16 kHz
            n = audio.shape[-1]
            if n != 512:
                audio = audio[-512:] if n > 512 else np.pad(audio, (0, 512 - n))

            try:
                tensor = torch.from_numpy(audio).float().unsqueeze(0)
                speech_prob = self.model(tensor, 16000).item()
            except Exception:
                return

            if speech_prob > self.threshold:
                silence_frames = 0
                if not is_speaking:
                    is_speaking = True
                    on_speech_start()
                buffer.append(audio.copy())
            else:
                if is_speaking:
                    silence_frames += 1
                    if silence_frames >= self.max_silence_frames:
                        is_speaking = False
                        silence_frames = 0
                        if buffer:
                            audio_data = np.concatenate(buffer)
                            pad = np.zeros(self.padding_samples, dtype=np.float32)
                            padded = np.concatenate([pad, audio_data, pad])
                            on_speech_end(padded)
                            buffer.clear()

        return callback


# ── ASR engine ─────────────────────────────────────────────────────────────────

class _ASREngine:
    """faster-whisper ASR — transcribes speech audio to text."""

    def __init__(self, model_name: str, language: str, compute_type: str) -> None:
        self.model_name = model_name
        self.language = language
        self.compute_type = compute_type
        self._model = None

    def load(self) -> None:
        from faster_whisper import WhisperModel
        # Use CPU + int8 for maximum speed on M1 (MPS not supported by faster-whisper)
        self._model = WhisperModel(
            self.model_name,
            device="cpu",
            compute_type=self.compute_type,
        )

    def transcribe(self, audio) -> str:
        import numpy as np
        if self._model is None:
            raise RuntimeError("ASR not loaded — call load() first")
        # faster-whisper expects float32 normalized to [-1, 1]
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32) / 32768.0
        segments, _ = self._model.transcribe(
            audio,
            language=self.language,
            beam_size=1,   # fastest — greedy decoding
            best_of=1,
        )
        return " ".join(s.text for s in segments).strip()

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None


# ── TTS engine ─────────────────────────────────────────────────────────────────

class _TTSEngine:
    """Piper TTS — faithful port of Specialist's streaming implementation.

    One Piper subprocess per speak() call. Full text written to stdin at once.
    PCM chunks are streamed directly to sounddevice.RawOutputStream as they
    arrive from Piper — audio starts before Piper finishes synthesizing.
    No sentence splitting, no producer threads, no gaps.
    """

    def __init__(self, config: VoiceIOConfig) -> None:
        self.config = config
        self._lock = threading.Lock()
        self._proc: Optional[subprocess.Popen] = None
        self._interrupted = threading.Event()

    def _build_cmd(self) -> list[str]:
        cmd = [
            self.config.piper_cmd,
            "--model", self.config.get_piper_model(),
            "--output-raw",
            "--length-scale", str(self.config.piper_length_scale),
            "--noise-scale", str(self.config.piper_noise_scale),
            "--noise-w", str(self.config.piper_noise_w),
        ]
        if self.config.piper_speaker_id is not None:
            cmd += ["--speaker", str(self.config.piper_speaker_id)]
        return cmd

    def warmup(self) -> None:
        """Pre-synthesize silence to load Piper model into memory."""
        list(self._stream_chunks("Hello."))
        print("[VoiceIO] TTS warmed up.", flush=True)

    def _stream_chunks(self, text: str, chunk_size: int = 8192):
        """Generator — yields raw PCM int16 chunks from Piper."""
        proc = None
        try:
            proc = subprocess.Popen(
                self._build_cmd(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            self._proc = proc
            proc.stdin.write(text.strip().encode("utf-8"))
        finally:
            # Always close stdin to signal EOF to Piper
            if proc and proc.stdin:
                proc.stdin.close()
        try:
            while True:
                chunk = proc.stdout.read(chunk_size)
                if not chunk:
                    break
                yield chunk
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
        except Exception as exc:
            print(f"[VoiceIO TTS error: {exc}]", flush=True)
        finally:
            # Kill process if still running (prevents zombie on abandoned generator)
            if proc and proc.poll() is None:
                proc.kill()
                try:
                    proc.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
            self._proc = None

    def speak(self, text: str) -> None:
        """Synthesize and stream text — audio starts as soon as first PCM arrives."""
        import numpy as np
        import sounddevice as sd

        if not text.strip() or not self.config.get_piper_model():
            return

        self._interrupted.clear()

        with self._lock:
            try:
                with sd.RawOutputStream(
                    samplerate=self.config.piper_sample_rate,
                    channels=1,
                    dtype="int16",
                ) as stream:
                    for chunk in self._stream_chunks(text):
                        if self._interrupted.is_set():
                            break
                        stream.write(np.frombuffer(chunk, dtype=np.int16))
            except Exception as exc:
                print(f"[VoiceIO playback error: {exc}]", flush=True)

    def interrupt(self) -> None:
        """Interrupt active TTS immediately."""
        import sounddevice as sd
        self._interrupted.set()
        try:
            sd.stop()
        except Exception:
            pass  # No active stream — safe to ignore
        proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.kill()

    def validate(self) -> None:
        if not self.config.get_piper_model():
            raise ValueError("VoiceIO: piper_model not configured. Run /voice config.")
        if not Path(self.config.get_piper_model()).expanduser().exists():
            raise FileNotFoundError(
                f"Piper model not found: {self.config.get_piper_model()}\n"
                "Update piper_model in ~/.eaudev/config.yml under voice_io."
            )


# ── VoiceIO — public API ───────────────────────────────────────────────────────

class VoiceIO:
    """Voice I/O mode for EauDev.

    Usage:
        voice = VoiceIO(config)
        voice.start()           # load models, open mic stream
        text = voice.listen()   # block until utterance → transcript
        voice.speak(text)       # TTS in background thread
        voice.stop()            # unload, close stream
    """

    def __init__(self, config: VoiceIOConfig | None = None) -> None:
        self.config = config or VoiceIOConfig()
        self._active = False
        self._vad: Optional[_VADEngine] = None
        self._asr: Optional[_ASREngine] = None
        self._tts: Optional[_TTSEngine] = None
        self._stream = None                     # sounddevice InputStream
        self._utterance_event = threading.Event()
        self._utterance_audio = None
        self._stop_event = threading.Event()    # signals listen() to abort
        self._speak_thread: Optional[threading.Thread] = None  # track TTS thread

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Load all models and open the microphone stream."""
        missing = check_dependencies()
        if missing:
            raise RuntimeError(
                "VoiceIO: missing dependencies:\n" +
                "\n".join(f"  • {m}" for m in missing)
            )

        import sounddevice as sd

        print("[VoiceIO] Loading VAD (Silero)...", flush=True)
        self._vad = _VADEngine(
            threshold=self.config.vad_threshold,
            min_silence_ms=self.config.vad_min_silence_ms,
            min_speech_ms=self.config.vad_min_speech_ms,
            padding_ms=self.config.vad_padding_ms,
        )

        print("[VoiceIO] Loading ASR (faster-whisper)...", flush=True)
        self._asr = _ASREngine(
            self.config.whisper_model,
            self.config.whisper_language,
            self.config.whisper_compute_type,
        )
        self._asr.load()

        if self.config.get_piper_model():
            print("[VoiceIO] Validating TTS (Piper)...", flush=True)
            self._tts = _TTSEngine(self.config)
            self._tts.validate()
            print("[VoiceIO] Warming up TTS...", flush=True)
            self._tts.warmup()
        else:
            print("[VoiceIO] TTS disabled — piper_model not configured.", flush=True)
            print("[VoiceIO] To enable TTS: set voice_io.piper_model in ~/.eaudev/config.yml", flush=True)
            print("[VoiceIO] Example: piper_model: /path/to/en_US-lessac-medium.onnx", flush=True)
            print("[VoiceIO] Install Piper: pip install piper-tts", flush=True)
            self._tts = None

        callback = self._vad.create_stream_callback(
            on_speech_start=self._on_speech_start,
            on_speech_end=self._on_speech_end,
        )

        self._stream = sd.InputStream(
            samplerate=self.config.sample_rate,
            channels=self.config.channels,
            blocksize=self.config.blocksize,
            dtype="float32",
            callback=callback,
        )
        self._stream.start()
        self._stop_event.clear()
        self._active = True
        print("[VoiceIO] Active — listening.", flush=True)

    def stop(self) -> None:
        """Close mic stream and unload all models."""
        self._active = False
        self._stop_event.set()          # unblocks listen() immediately
        self._utterance_event.set()     # unblocks any waiting listen() call
        if self._tts:
            self._tts.interrupt()       # kill active TTS
        if self._speak_thread and self._speak_thread.is_alive():
            self._speak_thread.join(timeout=1.0)
        if self._stream:
            try:
                self._stream.abort()    # non-blocking — don't wait for C thread
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._asr:
            self._asr.unload()
        self._vad = None
        self._tts = None
        self._stop_event.clear()
        print("[VoiceIO] Stopped.", flush=True)

    @property
    def active(self) -> bool:
        return self._active

    # ── Input ──────────────────────────────────────────────────────────────────

    def listen(self, timeout: float = 30.0) -> Optional[str]:
        """Block until a complete utterance is captured → return transcript.

        Waits for any previous TTS to finish before listening (prevents
        the microphone from picking up EauDev's own voice).

        Returns None on timeout or stop signal.
        Raises KeyboardInterrupt if stop() is called while listening.
        """
        if not self._active:
            return None

        # Wait for any in-progress TTS to finish before opening the mic
        if self._speak_thread and self._speak_thread.is_alive():
            self._speak_thread.join(timeout=10.0)

        self._stop_event.clear()
        self._utterance_event.clear()
        self._utterance_audio = None

        # Poll in 100ms intervals — responsive to stop() and KeyboardInterrupt
        elapsed = 0.0
        interval = 0.1
        while elapsed < timeout:
            if self._stop_event.is_set():
                raise KeyboardInterrupt
            if self._utterance_event.wait(timeout=interval):
                break
            elapsed += interval
        else:
            return None  # timeout

        if self._stop_event.is_set():
            raise KeyboardInterrupt
        if self._utterance_audio is None:
            return None

        try:
            text = self._asr.transcribe(self._utterance_audio)
            if self.config.print_transcript and text:
                print(f"\n[you said: {text}]", flush=True)
            return text if text else None
        except Exception as exc:
            print(f"[VoiceIO ASR error: {exc}]", flush=True)
            return None

    # ── Output ─────────────────────────────────────────────────────────────────

    def speak(self, text: str) -> None:
        """Speak text via Piper TTS in a background thread."""
        if not self._tts or not self.config.speak_responses:
            return
        # Join previous speak thread before starting new one
        if self._speak_thread and self._speak_thread.is_alive():
            self._speak_thread.join(timeout=0.1)
        self._speak_thread = threading.Thread(
            target=self._tts.speak, args=(text,), daemon=True
        )
        self._speak_thread.start()

    def speak_sync(self, text: str) -> None:
        """Speak text synchronously — blocks until audio finishes."""
        if not self._tts or not self.config.speak_responses:
            return
        self._tts.speak(text)

    def interrupt(self) -> None:
        """Interrupt active TTS playback immediately."""
        if self._tts:
            self._tts.interrupt()

    # ── Internal callbacks ─────────────────────────────────────────────────────

    def _on_speech_start(self) -> None:
        pass  # Could show mic-active indicator here

    def _on_speech_end(self, audio) -> None:
        self._utterance_audio = audio
        self._utterance_event.set()


# ── Singleton accessor ─────────────────────────────────────────────────────────

_voice_io_instance: Optional[VoiceIO] = None
_voice_io_lock = threading.Lock()


def get_voice_io() -> VoiceIO:
    """Return the global VoiceIO singleton, creating it if needed."""
    global _voice_io_instance
    with _voice_io_lock:
        if _voice_io_instance is None:
            _voice_io_instance = VoiceIO()
        return _voice_io_instance
