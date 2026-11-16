"""OpenAI services: speech-to-text, ChatGPT, and text-to-speech."""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator, Optional

from openai import AsyncOpenAI

from ..frames import (
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesFrame,
    LLMTextFrame,
)
from ..llm_context import LLMContext
from .base import LLMService, SegmentedSTTService, TTSService

logger = logging.getLogger(__name__)


class OpenAISTTService(SegmentedSTTService):
    """Transcribes each user utterance with OpenAI (gpt-4o-transcribe or whisper-1)."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-transcribe",
        language: Optional[str] = None,  # e.g. "en"; None lets the model detect it
        prompt: Optional[str] = None,  # words or names to help recognition
        base_url: Optional[str] = None,
        **kwargs,
    ):
        kwargs.setdefault("name", "OpenAISTT")
        super().__init__(**kwargs)
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._language = language
        self._prompt = prompt

    async def run_stt(self, wav: bytes) -> Optional[str]:
        extra = {}
        if self._language:
            extra["language"] = self._language
        if self._prompt:
            extra["prompt"] = self._prompt
        result = await self._client.audio.transcriptions.create(
            model=self._model,
            file=("audio.wav", wav, "audio/wav"),
            **extra,
        )
        return result.text


class OpenAILLMService(LLMService):
    """Streams replies from ChatGPT (OpenAI Chat Completions)."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        base_url: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        params: Optional[dict] = None,  # any other chat.completions.create arguments
        name: str = "OpenAILLM",
    ):
        super().__init__(name=name)
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._extra = {}
        if temperature is not None:
            self._extra["temperature"] = temperature
        if max_tokens is not None:
            self._extra["max_completion_tokens"] = max_tokens
        self._extra.update(params or {})

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMContextFrame):
            await self._generate(frame.context)
        elif isinstance(frame, LLMMessagesFrame):
            await self._generate(LLMContext(frame.messages))
        else:
            await self.push_frame(frame, direction)

    async def _generate(self, context: LLMContext):
        await self.push_frame(LLMFullResponseStartFrame())
        try:
            stream = await self._client.chat.completions.create(
                model=self._model,
                messages=context.get_messages(),
                stream=True,
                **self._extra,
            )
            async for chunk in stream:
                if not chunk.choices:
                    continue
                text = chunk.choices[0].delta.content
                if text:
                    await self.push_frame(LLMTextFrame(text=text))
        except asyncio.CancelledError:
            raise  # interrupted by the user: the reply is abandoned
        except Exception as e:
            logger.exception("ChatGPT request failed")
            await self.push_error(f"{self}: {e}")
        await self.push_frame(LLMFullResponseEndFrame())


class OpenAITTSService(TTSService):
    """Speaks text with OpenAI text-to-speech, streaming raw 24 kHz PCM."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini-tts",
        voice: str = "alloy",
        instructions: Optional[str] = None,  # tone of voice, gpt-4o-mini-tts only
        base_url: Optional[str] = None,
        **kwargs,
    ):
        kwargs.setdefault("name", "OpenAITTS")
        super().__init__(sample_rate=24000, **kwargs)
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._voice = voice
        self._instructions = instructions

    async def run_tts(self, text: str) -> AsyncIterator[bytes]:
        extra = {}
        if self._instructions:
            extra["instructions"] = self._instructions
        async with self._client.audio.speech.with_streaming_response.create(
            model=self._model,
            voice=self._voice,
            input=text,
            response_format="pcm",
            **extra,
        ) as response:
            async for chunk in response.iter_bytes(4096):
                yield chunk
