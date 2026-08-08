"""
Speech-to-text using NVIDIA Parakeet-TDT-0.6B-v2 via Riva gRPC (NVCF).

Uses the official nvidia-riva-client library to call:
  grpc.nvcf.nvidia.com:443

This is the canonical, stable interface documented at:
  https://build.nvidia.com/nvidia/parakeet-tdt-0_6b-v2

No local model is loaded — everything runs in NVIDIA's cloud.

Audio in: raw bytes (wav/flac/mp3/ogg) or a filesystem path.
Text out: transcription string.
"""

from __future__ import annotations

import io
import logging

import soundfile as sf
import riva.client

from app.core.config import settings

log = logging.getLogger(__name__)

# NVCF function-id for nvidia/parakeet-tdt-0.6b-v2
# Source: https://build.nvidia.com/nvidia/parakeet-tdt-0_6b-v2/api
_NVCF_FUNCTION_ID = "d3fe9151-442b-4204-a70d-5fcc597fd610"
_GRPC_ENDPOINT = "grpc.nvcf.nvidia.com:443"
_SAMPLE_RATE_HZ = 16_000  # Parakeet expects 16 kHz mono


def _to_wav_pcm16_mono(audio: bytes) -> bytes:
    """
    Re-encode arbitrary audio bytes to 16-bit mono WAV at 16 kHz.

    Parakeet TDT via Riva expects LINEAR_PCM 16-bit mono. We normalise
    here so that browser-recorded WebM/Opus, MP3, etc. all work reliably.
    """
    try:
        arr, sr = sf.read(io.BytesIO(audio), dtype="float32")
        # Mix down to mono
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        # Resample to 16 kHz if needed (simple decimation/interpolation via soundfile)
        if sr != _SAMPLE_RATE_HZ:
            import numpy as np
            from fractions import Fraction
            ratio = Fraction(_SAMPLE_RATE_HZ, sr).limit_denominator(100)
            # Use scipy if available, otherwise write at original sr (Riva can handle it)
            try:
                from scipy.signal import resample_poly
                arr = resample_poly(arr, ratio.numerator, ratio.denominator)
            except ImportError:
                pass  # fall through and send at original sr

        buf = io.BytesIO()
        sf.write(buf, arr, _SAMPLE_RATE_HZ, format="WAV", subtype="PCM_16")
        buf.seek(0)
        return buf.read()
    except Exception as exc:
        log.warning("Audio re-encoding failed (%s); passing raw bytes to Riva", exc)
        return audio


def _build_auth(api_key: str) -> riva.client.Auth:
    """Return a Riva Auth object pointed at the NVCF cloud endpoint."""
    return riva.client.Auth(
        uri=_GRPC_ENDPOINT,
        use_ssl=True,
        metadata_args=[
            ["function-id", _NVCF_FUNCTION_ID],
            ["authorization", f"Bearer {api_key}"],
        ],
    )


def transcribe(audio: bytes | str, sample_rate: int = 16_000) -> str:
    """
    Transcribe audio using the NVIDIA Parakeet TDT 0.6B v2 cloud model
    via the Riva gRPC interface on grpc.nvcf.nvidia.com:443.

    Parameters
    ----------
    audio:
        Either raw audio bytes (any common format: wav/webm/ogg/mp3/flac)
        or a filesystem path to an audio file.
    sample_rate:
        Hint used only when audio is already raw PCM bytes without a header.

    Returns
    -------
    str
        The transcribed text, or "" on an empty response.
    """
    api_key = settings.stt_api_key
    if not api_key:
        raise RuntimeError("STT_API_KEY is not set. Add it to your .env / Render env vars.")

    # --- Load audio bytes ---
    if isinstance(audio, str):
        with open(audio, "rb") as fh:
            raw = fh.read()
    else:
        raw = audio

    log.debug("Received %d raw audio bytes for transcription", len(raw))

    # --- Normalise to 16-bit mono WAV at 16 kHz ---
    wav_bytes = _to_wav_pcm16_mono(raw)
    log.debug("Sending %d bytes (normalised WAV) to Riva NVCF endpoint", len(wav_bytes))

    # --- Build Riva auth + ASR service ---
    auth = _build_auth(api_key)
    asr_service = riva.client.ASRService(auth)

    # --- Configure recognition ---
    # encoding, sample_rate_hertz, and audio_channel_count MUST be set explicitly;
    # Triton cannot auto-detect them even from a valid WAV header.
    config = riva.client.RecognitionConfig(
        encoding=riva.client.AudioEncoding.LINEAR_PCM,
        sample_rate_hertz=_SAMPLE_RATE_HZ,
        audio_channel_count=1,
        language_code="en-US",
        max_alternatives=1,
        enable_automatic_punctuation=True,
    )

    # --- Offline (batch) recognition call ---
    response = asr_service.offline_recognize(wav_bytes, config)

    if not response.results:
        log.warning("Riva returned no results for the submitted audio")
        return ""

    transcript = response.results[0].alternatives[0].transcript
    log.info("Transcription: %r", transcript)
    return transcript.strip()
