"""
Speech-to-text using NVIDIA Parakeet-TDT-0.6B-v2 via Riva gRPC (NVCF).

Uses the official nvidia-riva-client library to call:
  grpc.nvcf.nvidia.com:443

This is the canonical, stable interface documented at:
  https://build.nvidia.com/nvidia/parakeet-tdt-0_6b-v2

No local model is loaded — everything runs in NVIDIA's cloud.

Audio in: raw bytes (any format: WebM/Opus, WAV, FLAC, MP3, OGG) or a path.
Text out: transcription string.

Decoding pipeline
-----------------
1. PyAV  — handles WebM/Opus from browser MediaRecorder and virtually all
            container/codec combos; bundles its own FFmpeg so no system dep.
2. soundfile — fallback for WAV / FLAC / OGG-Vorbis / AIFF.
Both strategies always produce 16-bit mono WAV at 16 kHz before sending to Riva.
Raw bytes are NEVER forwarded to Riva undecoded.
"""

from __future__ import annotations

import io
import logging

import numpy as np
import soundfile as sf
import riva.client

from app.core.config import settings

log = logging.getLogger(__name__)

# NVCF function-id for nvidia/parakeet-tdt-0.6b-v2
# Source: https://build.nvidia.com/nvidia/parakeet-tdt-0_6b-v2/api
_NVCF_FUNCTION_ID = "d3fe9151-442b-4204-a70d-5fcc597fd610"
_GRPC_ENDPOINT = "grpc.nvcf.nvidia.com:443"
_SAMPLE_RATE_HZ = 16_000  # Parakeet expects 16 kHz mono


# ---------------------------------------------------------------------------
# Audio normalisation helpers
# ---------------------------------------------------------------------------

def _resample(samples: np.ndarray, src_rate: int) -> np.ndarray:
    """Resample a 1-D float32 array from src_rate to _SAMPLE_RATE_HZ."""
    if src_rate == _SAMPLE_RATE_HZ:
        return samples
    try:
        from fractions import Fraction
        from scipy.signal import resample_poly
        r = Fraction(_SAMPLE_RATE_HZ, src_rate).limit_denominator(100)
        return resample_poly(samples, r.numerator, r.denominator).astype(np.float32)
    except ImportError:
        log.warning("scipy not available; skipping resample (src=%d Hz)", src_rate)
        return samples


def _to_wav_bytes(samples: np.ndarray) -> bytes:
    """Write a 1-D float32 mono array to 16-bit WAV bytes at _SAMPLE_RATE_HZ."""
    buf = io.BytesIO()
    sf.write(buf, samples, _SAMPLE_RATE_HZ, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


def _decode_via_pyav(audio: bytes) -> np.ndarray:
    """
    Decode any container/codec (WebM/Opus, MP4/AAC, MP3, …) to float32 mono
    using PyAV's bundled FFmpeg — no system libraries required.
    Returns a 1-D float32 numpy array at the source sample rate.
    Raises on failure so the caller can fall through to the next strategy.
    """
    import av as pyav  # noqa: PLC0415

    container = pyav.open(io.BytesIO(audio))
    audio_stream = next(
        (s for s in container.streams if s.type == "audio"), None
    )
    if audio_stream is None:
        raise ValueError("No audio stream found in container")

    src_rate: int = (
        audio_stream.codec_context.sample_rate
        or audio_stream.sample_rate
        or _SAMPLE_RATE_HZ
    )
    chunks: list[np.ndarray] = []
    for frame in container.decode(audio_stream):
        # force float32 planar → shape (n_channels, n_samples)
        arr = frame.to_ndarray(format="fltp")
        chunks.append(arr.mean(axis=0))  # mix-down to mono

    if not chunks:
        raise ValueError("PyAV decoded zero audio frames")

    samples = np.concatenate(chunks).astype(np.float32)
    log.debug("PyAV: decoded %d samples at %d Hz", len(samples), src_rate)
    return _resample(samples, src_rate)


def _decode_via_soundfile(audio: bytes) -> np.ndarray:
    """
    Decode WAV / FLAC / OGG-Vorbis / AIFF via libsndfile (soundfile).
    Returns a 1-D float32 numpy array resampled to _SAMPLE_RATE_HZ.
    Raises on failure.
    """
    arr, sr = sf.read(io.BytesIO(audio), dtype="float32")
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    log.debug("soundfile: decoded %d samples at %d Hz", len(arr), sr)
    return _resample(arr, sr)


def _to_wav_pcm16_mono(audio: bytes) -> bytes:
    """
    Convert arbitrary audio bytes to 16-bit mono WAV at 16 kHz.
    Tries PyAV first, then soundfile; raises RuntimeError if both fail.
    Raw bytes are NEVER passed through to Riva undecoded.
    """
    av_exc: Exception | None = None
    sf_exc: Exception | None = None

    # Strategy 1 — PyAV (WebM/Opus, MP3, MP4, …)
    try:
        samples = _decode_via_pyav(audio)
        return _to_wav_bytes(samples)
    except Exception as exc:
        av_exc = exc
        log.warning("PyAV decode failed: %s", exc)

    # Strategy 2 — soundfile (WAV, FLAC, OGG-Vorbis, AIFF)
    try:
        samples = _decode_via_soundfile(audio)
        return _to_wav_bytes(samples)
    except Exception as exc:
        sf_exc = exc
        log.warning("soundfile decode failed: %s", exc)

    raise RuntimeError(
        f"Audio decode failed — PyAV: {av_exc!r}; soundfile: {sf_exc!r}"
    )


# ---------------------------------------------------------------------------
# gRPC helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def transcribe(audio: bytes | str, sample_rate: int = 16_000) -> str:
    """
    Transcribe audio using the NVIDIA Parakeet TDT 0.6B v2 cloud model
    via the Riva gRPC interface on grpc.nvcf.nvidia.com:443.

    Parameters
    ----------
    audio:
        Either raw audio bytes (WebM/Opus, WAV, FLAC, MP3, OGG, …)
        or a filesystem path to an audio file.
    sample_rate:
        Unused — sample rate is always read from the audio content.

    Returns
    -------
    str
        The transcribed text, or "" on an empty response.
    """
    api_key = settings.stt_api_key
    if not api_key:
        raise RuntimeError("STT_API_KEY is not set. Add it to your Render env vars.")

    # --- Load bytes ---
    if isinstance(audio, str):
        with open(audio, "rb") as fh:
            raw = fh.read()
    else:
        raw = audio

    log.debug("Received %d raw audio bytes for transcription", len(raw))

    # --- Decode & normalise to 16-bit mono WAV @ 16 kHz ---
    wav_bytes = _to_wav_pcm16_mono(raw)
    log.debug("Sending %d normalised WAV bytes to Riva NVCF", len(wav_bytes))

    # --- Riva gRPC call ---
    auth = _build_auth(api_key)
    asr_service = riva.client.ASRService(auth)

    # encoding, sample_rate_hertz, and audio_channel_count MUST be explicit;
    # Triton cannot auto-detect them even from a valid WAV header.
    config = riva.client.RecognitionConfig(
        encoding=riva.client.AudioEncoding.LINEAR_PCM,
        sample_rate_hertz=_SAMPLE_RATE_HZ,
        audio_channel_count=1,
        language_code="en-US",
        max_alternatives=1,
        enable_automatic_punctuation=True,
    )

    response = asr_service.offline_recognize(wav_bytes, config)

    if not response.results:
        log.warning("Riva returned no results for the submitted audio")
        return ""

    transcript = response.results[0].alternatives[0].transcript
    log.info("Transcription: %r", transcript)
    return transcript.strip()
