"""
Speech-to-text using NVIDIA NIM Parakeet-TDT-0.6B-v2 (cloud API).

Sends audio to the NVIDIA NIM endpoint at:
  https://integrate.api.nvidia.com/v1/audio/transcriptions

The endpoint is OpenAI-compatible (multipart/form-data upload).
No local model is loaded — everything runs in NVIDIA's cloud.

Audio in: raw bytes (wav/flac/mp3/ogg) or a file path.
Text out: transcription string.
"""

from __future__ import annotations

import io
import logging

import httpx
import soundfile as sf

from app.core.config import settings

log = logging.getLogger(__name__)

_NVIDIA_STT_URL = "https://integrate.api.nvidia.com/v1/audio/transcriptions"
_MODEL = "nvidia/parakeet-tdt-0.6b-v2"


def _ensure_wav_bytes(audio: bytes) -> tuple[bytes, str]:
    """
    Re-encode audio bytes to WAV PCM-16 if necessary.
    Returns (wav_bytes, filename_hint).
    """
    try:
        arr, sr = sf.read(io.BytesIO(audio), dtype="float32")
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        buf = io.BytesIO()
        sf.write(buf, arr, sr, format="WAV", subtype="PCM_16")
        buf.seek(0)
        return buf.read(), "audio.wav"
    except Exception:
        # Pass through as-is; let the API handle the codec
        return audio, "audio.wav"


def transcribe(audio: bytes | str, sample_rate: int = 16000) -> str:
    """
    Transcribe audio using the NVIDIA NIM cloud API.

    Parameters
    ----------
    audio:
        Either raw audio bytes (any common format) or a filesystem path to
        an audio file.
    sample_rate:
        Ignored when audio is bytes (sr is read from the file header).
        Only used when audio is a raw numpy array (not supported here).

    Returns
    -------
    str
        The transcribed text, or "" on empty response.
    """
    api_key = settings.stt_api_key
    if not api_key:
        raise RuntimeError("STT_API_KEY is not set. Add it to your .env file.")

    headers = {"Authorization": f"Bearer {api_key}"}

    if isinstance(audio, str):
        # File path — read it
        with open(audio, "rb") as fh:
            raw = fh.read()
        fname = audio.split("/")[-1] or "audio.wav"
        audio_bytes = raw
    else:
        audio_bytes, fname = _ensure_wav_bytes(audio)

    files = {"file": (fname, audio_bytes, "audio/wav")}
    data = {"model": _MODEL}

    log.debug("Sending %d bytes of audio to NVIDIA NIM STT", len(audio_bytes))

    with httpx.Client(timeout=120.0) as client:
        r = client.post(_NVIDIA_STT_URL, headers=headers, files=files, data=data)

    if r.status_code != 200:
        log.error("NVIDIA STT error %s: %s", r.status_code, r.text)
        r.raise_for_status()

    resp = r.json()

    # OpenAI-compatible response: {"text": "..."}
    text = resp.get("text", "")
    if not text and "results" in resp:
        # Some NVIDIA endpoints wrap in a results list
        results = resp["results"]
        if results:
            text = results[0].get("transcript", "")

    return (text or "").strip()
