"""Small audio helpers: read WAV bytes, downmix and resample 16-bit PCM."""

from __future__ import annotations

import struct
from typing import Tuple

import numpy as np


def wav_to_pcm(data: bytes) -> Tuple[bytes, int, int]:
    """Return (pcm, sample_rate, num_channels) from 16-bit WAV bytes.

    Reads the chunks by hand because streamed WAVs often carry bogus sizes
    that Python's wave module rejects.
    """
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("not a WAV file")
    sample_rate, num_channels, bits = 24000, 1, 16
    pos = 12
    while pos + 8 <= len(data):
        chunk_id = data[pos : pos + 4]
        size = struct.unpack("<I", data[pos + 4 : pos + 8])[0]
        body = pos + 8
        if chunk_id == b"fmt ":
            _, num_channels, sample_rate, _, _, bits = struct.unpack("<HHIIHH", data[body : body + 16])
        elif chunk_id == b"data":
            end = len(data) if size in (0, 0xFFFFFFFF) or body + size > len(data) else body + size
            if bits != 16:
                raise ValueError(f"expected 16-bit WAV, got {bits}-bit")
            return data[body:end], sample_rate, num_channels
        pos = body + size + (size & 1)
    raise ValueError("WAV has no data chunk")


def to_mono(pcm: bytes, num_channels: int) -> bytes:
    if num_channels == 1:
        return pcm
    samples = np.frombuffer(pcm, dtype=np.int16).reshape(-1, num_channels)
    return samples.mean(axis=1).astype(np.int16).tobytes()


def resample(pcm: bytes, from_rate: int, to_rate: int) -> bytes:
    """Linear-interpolation resampling of mono 16-bit PCM. Good enough for speech."""
    if from_rate == to_rate or not pcm:
        return pcm
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    count = int(round(len(samples) * to_rate / from_rate))
    positions = np.linspace(0, len(samples) - 1, num=count)
    out = np.interp(positions, np.arange(len(samples)), samples)
    return np.clip(out, -32768, 32767).astype(np.int16).tobytes()
