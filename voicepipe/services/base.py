"""Base classes for speech-to-text, LLM and text-to-speech services."""

from __future__ import annotations

import io
import logging
import re
import wave
from datetime import datetime, timezone
from typing import AsyncIterator, Optional

from ..frames import (
    InputAudioRawFrame,
    LLMFullResponseEndFrame,
    LLMTextFrame,
    StartFrame,
    StartInterruptionFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from ..llm_context import LLMContext, LLMContextAggregatorPair
from ..processor import FrameProcessor

logger = logging.getLogger(__name__)


def pcm_to_wav(pcm: bytes, sample_rate: int, num_channels: int = 1) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(num_channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return buffer.getvalue()


class SegmentedSTTService(FrameProcessor):
    """Buffers audio while the user talks, then transcribes the whole utterance.

    Subclasses implement run_stt(wav_bytes) -> text.
    """

    def __init__(self, *, preroll_secs: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self._preroll_secs = preroll_secs  # keeps the start of words VAD reacts late to
        self._sample_rate = 16000
        self._buffer = bytearray()
        self._user_speaking = False

    async def run_stt(self, wav: bytes) -> Optional[str]:
        raise NotImplementedError

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, StartFrame):
            self._sample_rate = frame.audio_in_sample_rate
            await self.push_frame(frame, direction)
        elif isinstance(frame, InputAudioRawFrame):
            self._buffer.extend(frame.audio)
            if not self._user_speaking:
                keep = int(self._sample_rate * self._preroll_secs) * 2
                if len(self._buffer) > keep:
                    del self._buffer[: len(self._buffer) - keep]
        elif isinstance(frame, UserStartedSpeakingFrame):
            self._user_speaking = True
            await self.push_frame(frame, direction)
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._user_speaking = False
            audio = bytes(self._buffer)
            self._buffer.clear()
            await self.push_frame(frame, direction)
            # Transcribe in the background so incoming audio keeps flowing.
            self.create_task(self._transcribe(audio), name=f"{self}::transcribe")
        else:
            await self.push_frame(frame, direction)

    async def _transcribe(self, pcm: bytes):
        try:
            text = await self.run_stt(pcm_to_wav(pcm, self._sample_rate))
        except Exception as e:
            logger.exception("Transcription failed")
            await self.push_error(f"{self}: {e}")
            return
        if text and text.strip():
            logger.info("User: %s", text.strip())
            timestamp = datetime.now(timezone.utc).isoformat()
            await self.push_frame(TranscriptionFrame(text=text.strip(), timestamp=timestamp))


class LLMService(FrameProcessor):
    """Base for LLM services. Subclasses handle LLMContextFrame."""

    def create_context_aggregator(self, context: LLMContext) -> LLMContextAggregatorPair:
        return LLMContextAggregatorPair(context)


# End of a sentence: punctuation (plus closing quotes/brackets) followed by space.
_SENTENCE_END = re.compile(r"[.!?;:。！？]+[\"')\]]*\s+|\n+")


class TTSService(FrameProcessor):
    """Turns streamed LLM text into audio, one sentence at a time.

    Speaking each sentence as soon as it's complete means the bot starts
    talking before the LLM has finished its reply.
    Subclasses implement run_tts(text) as an async generator of PCM bytes.
    """

    def __init__(self, *, sample_rate: int = 24000, **kwargs):
        super().__init__(**kwargs)
        self.sample_rate = sample_rate
        self._text = ""
        self._speaking = False

    async def run_tts(self, text: str) -> AsyncIterator[bytes]:
        raise NotImplementedError
        yield b""  # pragma: no cover  (makes this an async generator)

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMTextFrame):
            self._text += frame.text
            await self._say_complete_sentences()
        elif isinstance(frame, LLMFullResponseEndFrame):
            await self._say(self._text)
            self._text = ""
            await self._finish()
            await self.push_frame(frame, direction)
        elif isinstance(frame, TTSSpeakFrame):
            await self._say(frame.text)
            await self._finish()
        elif isinstance(frame, StartInterruptionFrame):
            self._text = ""
            self._speaking = False
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)

    async def _say_complete_sentences(self):
        while True:
            match = _SENTENCE_END.search(self._text)
            if not match:
                return
            sentence, self._text = self._text[: match.end()], self._text[match.end() :]
            await self._say(sentence)

    async def _say(self, text: str):
        text = text.strip()
        if not text:
            return
        if not self._speaking:
            self._speaking = True
            await self.push_frame(TTSStartedFrame())
        logger.info("Bot: %s", text)
        leftover = b""
        async for chunk in self.run_tts(text):
            data = leftover + chunk
            whole = len(data) - (len(data) % 2)  # never split a 16-bit sample
            leftover = data[whole:]
            if whole:
                await self.push_frame(TTSAudioRawFrame(audio=data[:whole], sample_rate=self.sample_rate))
        await self.push_frame(TTSTextFrame(text=text))

    async def _finish(self):
        if self._speaking:
            self._speaking = False
            await self.push_frame(TTSStoppedFrame())
