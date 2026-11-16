"""Voice activity detection: decides when the user starts and stops talking."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


class VADState(Enum):
    QUIET = 1
    STARTING = 2  # heard speech, not long enough to count yet
    SPEAKING = 3
    STOPPING = 4  # heard silence, not long enough to count yet


@dataclass
class VADParams:
    start_secs: float = 0.2  # speech needed before the user "started speaking"
    stop_secs: float = 0.8  # silence needed before the user "stopped speaking"


class VADAnalyzer:
    """Turns a stream of audio chunks into speaking / not-speaking states.

    Subclasses only decide whether one chunk contains speech (is_speech).
    """

    def __init__(self, *, sample_rate: Optional[int] = None, params: Optional[VADParams] = None):
        self.sample_rate = sample_rate
        self.params = params or VADParams()
        self._state = VADState.QUIET
        self._speech_secs = 0.0
        self._silence_secs = 0.0

    def set_sample_rate(self, sample_rate: int):
        self.sample_rate = sample_rate

    def is_speech(self, audio: bytes) -> bool:
        raise NotImplementedError

    def analyze(self, audio: bytes) -> VADState:
        if not self.sample_rate:
            raise RuntimeError("VADAnalyzer needs a sample rate")
        secs = len(audio) / 2 / self.sample_rate
        if self.is_speech(audio):
            self._silence_secs = 0.0
            self._speech_secs += secs
            if self._state in (VADState.QUIET, VADState.STARTING):
                started = self._speech_secs >= self.params.start_secs
                self._state = VADState.SPEAKING if started else VADState.STARTING
            else:
                self._state = VADState.SPEAKING
        elif self._state in (VADState.SPEAKING, VADState.STOPPING):
            self._silence_secs += secs
            if self._silence_secs >= self.params.stop_secs:
                self._state = VADState.QUIET
                self._speech_secs = self._silence_secs = 0.0
            else:
                self._state = VADState.STOPPING
        else:
            self._state = VADState.QUIET
            self._speech_secs = 0.0
        return self._state


class EnergyVADAnalyzer(VADAnalyzer):
    """Loudness-based VAD with an adaptive noise floor. No model download needed.

    Use headphones: with speakers, the mic hears the bot and will interrupt it.
    """

    def __init__(self, *, min_volume: float = 0.015, noise_ratio: float = 3.0, **kwargs):
        super().__init__(**kwargs)
        self.min_volume = min_volume  # RMS on a 0..1 scale
        self.noise_ratio = noise_ratio  # speech must be this many times the noise floor
        self._noise_floor = min_volume / noise_ratio

    def is_speech(self, audio: bytes) -> bool:
        samples = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        if samples.size == 0:
            return False
        rms = float(np.sqrt(np.mean(samples * samples)))
        threshold = max(self.min_volume, self._noise_floor * self.noise_ratio)
        speech = rms > threshold
        if not speech:  # learn the room's background noise level slowly
            self._noise_floor = 0.95 * self._noise_floor + 0.05 * rms
        return speech
