"""
Speech-to-text using NVIDIA Parakeet-TDT-0.6B-v2 via Riva gRPC (NVCF).

Endpoint : grpc.nvcf.nvidia.com:443
Function : d3fe9151-442b-4204-a70d-5fcc597fd610
Docs     : https://build.nvidia.com/nvidia/parakeet-tdt-0_6b-v2

Audio decoding
--------------
PyAV (bundled FFmpeg, no system dependencies) with av.AudioResampler:
  - Decodes any container/codec the browser sends (WebM/Opus, WAV, …).
  - Resamples + mixes down to 16 kHz int16 mono in one step via libswresample.
  - Raw int16 bytes (no WAV header) are sent to Riva with encoding=LINEAR_PCM.
"""

from __future__ import annotations

import io
import logging

import av
import numpy as np
import riva.client

from app.core.config import settings

log = logging.getLogger(__name__)

_NVCF_FUNCTION_ID = "d3fe9151-442b-4204-a70d-5fcc597fd610"
_GRPC_ENDPOINT    = "grpc.nvcf.nvidia.com:443"
_SAMPLE_RATE_HZ   = 16_000  # Parakeet expects 16 kHz mono int16


# ---------------------------------------------------------------------------
# Audio decoding
# ---------------------------------------------------------------------------

def _to_raw_pcm16(audio: bytes) -> bytes:
    """
    Decode any audio container/codec → raw int16 mono PCM at 16 kHz.

    Uses av.AudioResampler which internally calls libswresample to handle:
      - Codec decoding  (Opus, Vorbis, AAC, MP3, …)
      - Format conversion  (float32 planar → signed int16 packed)
      - Channel down-mix  (stereo / surround → mono)
      - Sample-rate conversion  (48 kHz / 44.1 kHz / … → 16 kHz)

    Returns raw bytes with NO header — Riva consumes them directly when
    RecognitionConfig.encoding = LINEAR_PCM.

    Raises ValueError on decode failure so the caller gets a clean 500
    instead of sending garbage bytes to Riva.
    """
    container = av.open(io.BytesIO(audio))

    audio_streams = list(container.streams.audio)
    if not audio_streams:
        raise ValueError("Uploaded file contains no audio stream")

    resampler = av.AudioResampler(
        format="s16",           # signed 16-bit packed  → dtype int16
        layout="mono",          # down-mix all channels to mono
        rate=_SAMPLE_RATE_HZ,   # resample to 16 kHz
    )

    chunks: list[np.ndarray] = []

    for frame in container.decode(audio_streams[0]):
        # resample() may buffer internally; always iterate the returned list
        for out_frame in resampler.resample(frame):
            # to_ndarray() on s16 mono returns int16, shape (1, n_samples)
            chunks.append(out_frame.to_ndarray().flatten())

    # Flush any samples the resampler is still holding
    for out_frame in resampler.resample(None):
        chunks.append(out_frame.to_ndarray().flatten())

    if not chunks:
        raise ValueError("PyAV decoded zero audio frames from uploaded file")

    pcm = np.concatenate(chunks).astype(np.int16)
    duration_s = len(pcm) / _SAMPLE_RATE_HZ
    log.info("Decoded %.2f s of audio (%d int16 samples @ %d Hz)",
             duration_s, len(pcm), _SAMPLE_RATE_HZ)
    return pcm.tobytes()


# ---------------------------------------------------------------------------
# Riva gRPC helpers
# ---------------------------------------------------------------------------

def _build_auth(api_key: str) -> riva.client.Auth:
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
    Transcribe audio using NVIDIA Parakeet TDT 0.6B v2 (Riva gRPC / NVCF).

    Parameters
    ----------
    audio:
        Raw audio bytes (WebM/Opus, WAV, FLAC, MP3, OGG, …) or a file path.
    sample_rate:
        Ignored — sample rate is always read from the container by PyAV.

    Returns
    -------
    str
        Transcribed text, or "" on an empty Riva response.
    """
    api_key = settings.stt_api_key
    if not api_key:
        raise RuntimeError("STT_API_KEY is not set in Render environment variables.")

    # --- Load bytes ---
    if isinstance(audio, str):
        with open(audio, "rb") as fh:
            raw = fh.read()
    else:
        raw = audio

    log.info("STT: received %d raw bytes", len(raw))

    # --- Decode to raw int16 PCM via PyAV ---
    pcm_bytes = _to_raw_pcm16(raw)
    log.info("STT: sending %d PCM bytes to Riva NVCF", len(pcm_bytes))

    # --- Riva gRPC call ---
    auth = _build_auth(api_key)
    asr  = riva.client.ASRService(auth)

    # encoding, sample_rate_hertz, audio_channel_count MUST be explicit.
    # Triton's audio decoder does not auto-detect these values.
    config = riva.client.RecognitionConfig(
        encoding=riva.client.AudioEncoding.LINEAR_PCM,
        sample_rate_hertz=_SAMPLE_RATE_HZ,
        audio_channel_count=1,
        language_code="en-US",
        max_alternatives=1,
        enable_automatic_punctuation=True,
    )

    response = asr.offline_recognize(pcm_bytes, config)

    if not response.results:
        log.warning("Riva returned no results")
        return ""

    transcript = response.results[0].alternatives[0].transcript
    log.info("Transcription: %r", transcript)
    return transcript.strip()
