"""Groq services: Whisper speech-to-text, gpt-oss chat, and Orpheus text-to-speech.

Groq's API is OpenAI-compatible, so these reuse the OpenAI services with
Groq's address and models. One free key (GROQ_API_KEY) covers all three.
"""

from __future__ import annotations

import os
from typing import AsyncIterator, List, Optional

from openai import AsyncOpenAI

from ..audio import resample, to_mono, wav_to_pcm
from .base import TTSService
from .openai_service import OpenAILLMService, OpenAISTTService

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Orpheus reads at most 200 characters per request.
ORPHEUS_MAX_CHARS = 200
ORPHEUS_VOICES = ("troy", "austin", "daniel", "hannah", "diana", "autumn")


def _api_key(api_key: Optional[str]) -> Optional[str]:
    return api_key or os.getenv("GROQ_API_KEY")


class GroqSTTService(OpenAISTTService):
    """Speech-to-text with Whisper on Groq."""

    def __init__(self, *, api_key: Optional[str] = None, model: str = "whisper-large-v3-turbo", **kwargs):
        kwargs.setdefault("name", "GroqSTT")
        super().__init__(api_key=_api_key(api_key), model=model, base_url=GROQ_BASE_URL, **kwargs)


class GroqLLMService(OpenAILLMService):
    """Chat with an LLM on Groq. Defaults to OpenAI's open gpt-oss model.

    gpt-oss thinks before it answers. For a voice agent the thinking is kept
    short (reasoning_effort="low") and left out of the reply so it isn't spoken.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = "openai/gpt-oss-20b",
        params: Optional[dict] = None,
        **kwargs,
    ):
        merged = {}
        if model.startswith("openai/gpt-oss"):
            merged = {"reasoning_effort": "low", "extra_body": {"include_reasoning": False}}
        merged.update(params or {})
        kwargs.setdefault("name", "GroqLLM")
        super().__init__(
            api_key=_api_key(api_key), model=model, base_url=GROQ_BASE_URL, params=merged, **kwargs
        )


class GroqTTSService(TTSService):
    """Text-to-speech with Orpheus on Groq.

    Orpheus returns a WAV file per request and reads at most 200 characters,
    so long sentences are split at word boundaries. `direction` is an optional
    vocal direction such as "[cheerful]" or "[warm]" put before each piece.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = "canopylabs/orpheus-v1-english",
        voice: str = "troy",
        direction: Optional[str] = None,
        sample_rate: int = 24000,
        **kwargs,
    ):
        kwargs.setdefault("name", "GroqTTS")
        super().__init__(sample_rate=sample_rate, **kwargs)
        self._client = AsyncOpenAI(api_key=_api_key(api_key), base_url=GROQ_BASE_URL)
        self._model = model
        self._voice = voice
        self._direction = f"{direction.strip()} " if direction else ""

    async def run_tts(self, text: str) -> AsyncIterator[bytes]:
        for piece in split_text(text, ORPHEUS_MAX_CHARS - len(self._direction)):
            async with self._client.audio.speech.with_streaming_response.create(
                model=self._model,
                voice=self._voice,
                input=self._direction + piece,
                response_format="wav",
            ) as response:
                wav = await response.read()
            pcm, rate, channels = wav_to_pcm(wav)
            yield resample(to_mono(pcm, channels), rate, self.sample_rate)


def split_text(text: str, max_chars: int) -> List[str]:
    """Split text into pieces of at most max_chars, breaking between words."""
    pieces: List[str] = []
    current = ""
    for word in text.split():
        while len(word) > max_chars:  # a single absurdly long "word"
            if current:
                pieces.append(current)
                current = ""
            pieces.append(word[:max_chars])
            word = word[max_chars:]
        candidate = f"{current} {word}" if current else word
        if len(candidate) > max_chars:
            pieces.append(current)
            current = word
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces
