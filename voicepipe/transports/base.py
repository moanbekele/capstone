"""Base input/output transports: where audio enters and leaves the pipeline."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

from ..frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    InputAudioRawFrame,
    OutputAudioRawFrame,
    StartFrame,
    StartInterruptionFrame,
    StopInterruptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from ..processor import FrameDirection, FrameProcessor
from ..vad import VADAnalyzer, VADState

logger = logging.getLogger(__name__)

# How long the speaker must be silent before the bot counts as "stopped speaking".
BOT_STOPPED_SPEAKING_SECS = 0.35


@dataclass
class TransportParams:
    audio_in_enabled: bool = True
    audio_out_enabled: bool = True
    audio_in_sample_rate: int = 16000
    audio_out_sample_rate: int = 24000  # OpenAI TTS returns 24 kHz PCM
    audio_in_chunk_ms: int = 20
    audio_out_chunk_ms: int = 40  # small chunks so interruptions cut in fast
    vad_analyzer: Optional[VADAnalyzer] = None


class BaseInputTransport(FrameProcessor):
    """Turns raw microphone audio into frames and runs VAD to find turns.

    Subclasses open the device in start() and feed bytes into
    self._audio_in_queue (16-bit mono PCM).
    """

    def __init__(self, params: TransportParams, **kwargs):
        super().__init__(**kwargs)
        self._params = params
        self._audio_in_queue: asyncio.Queue = asyncio.Queue()
        self._audio_task: Optional[asyncio.Task] = None
        self._user_speaking = False

    async def start(self, frame: StartFrame):
        if self._params.vad_analyzer:
            self._params.vad_analyzer.set_sample_rate(self._params.audio_in_sample_rate)
        self._audio_task = self.create_task(self._audio_loop(), name=f"{self}::audio")

    async def stop(self):
        await self.cancel_task(self._audio_task)
        self._audio_task = None

    async def cleanup(self):
        await self.stop()
        await super().cleanup()

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, StartFrame):
            await self.push_frame(frame, direction)
            if self._params.audio_in_enabled:
                await self.start(frame)
        elif isinstance(frame, (EndFrame, CancelFrame)):
            await self.stop()
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)

    async def _audio_loop(self):
        vad = self._params.vad_analyzer
        while True:
            audio = await self._audio_in_queue.get()
            if vad:
                await self._handle_vad(vad.analyze(audio))
            await self.push_frame(
                InputAudioRawFrame(audio=audio, sample_rate=self._params.audio_in_sample_rate)
            )

    async def _handle_vad(self, state: VADState):
        if state == VADState.SPEAKING and not self._user_speaking:
            self._user_speaking = True
            logger.info("User started speaking")
            await self.push_frame(UserStartedSpeakingFrame())
            if self.interruptions_allowed:
                await self.push_frame(StartInterruptionFrame())
        elif state == VADState.QUIET and self._user_speaking:
            self._user_speaking = False
            logger.info("User stopped speaking")
            await self.push_frame(UserStoppedSpeakingFrame())
            if self.interruptions_allowed:
                await self.push_frame(StopInterruptionFrame())


class BaseOutputTransport(FrameProcessor):
    """Plays audio frames and reports when the bot starts and stops speaking.

    Subclasses open the device in start() and implement write_audio().
    """

    def __init__(self, params: TransportParams, **kwargs):
        super().__init__(**kwargs)
        self._params = params
        self._audio_out_queue: asyncio.Queue = asyncio.Queue()
        self._audio_task: Optional[asyncio.Task] = None
        self._bot_speaking = False
        self._chunk_bytes = int(params.audio_out_sample_rate * params.audio_out_chunk_ms / 1000) * 2

    async def start(self, frame: StartFrame):
        self._audio_task = self.create_task(self._audio_loop(), name=f"{self}::audio")

    async def stop(self):
        await self.cancel_task(self._audio_task)
        self._audio_task = None

    async def cleanup(self):
        await self.stop()
        await super().cleanup()

    async def write_audio(self, audio: bytes):
        """Play one chunk of 16-bit mono PCM. Should return once it is written."""
        raise NotImplementedError

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, StartFrame):
            if self._params.audio_out_enabled:
                await self.start(frame)
            await self.push_frame(frame, direction)
        elif isinstance(frame, CancelFrame):
            await self.stop()
            await self.push_frame(frame, direction)
        elif isinstance(frame, EndFrame):
            await self._wait_until_played()
            await self.stop()
            await self.push_frame(frame, direction)
        elif isinstance(frame, StartInterruptionFrame):
            await self._clear_audio()
            await self.push_frame(frame, direction)
        elif isinstance(frame, OutputAudioRawFrame):
            await self._queue_audio(frame)
        else:
            await self.push_frame(frame, direction)

    async def _queue_audio(self, frame: OutputAudioRawFrame):
        if frame.sample_rate != self._params.audio_out_sample_rate:
            logger.warning(
                "Audio at %d Hz but the speaker is set to %d Hz",
                frame.sample_rate,
                self._params.audio_out_sample_rate,
            )
        audio = frame.audio
        for i in range(0, len(audio), self._chunk_bytes):
            await self._audio_out_queue.put(audio[i : i + self._chunk_bytes])

    async def _clear_audio(self):
        """The user interrupted: stop playing and forget queued audio."""
        if self._audio_task is None:
            return
        await self.cancel_task(self._audio_task)
        self._audio_out_queue = asyncio.Queue()
        self._audio_task = self.create_task(self._audio_loop(), name=f"{self}::audio")
        if self._bot_speaking:
            await self._bot_stopped_speaking()

    async def _wait_until_played(self):
        while self._audio_task is not None and not self._audio_out_queue.empty():
            await asyncio.sleep(0.05)

    async def _audio_loop(self):
        while True:
            try:
                chunk = await asyncio.wait_for(
                    self._audio_out_queue.get(), timeout=BOT_STOPPED_SPEAKING_SECS
                )
            except asyncio.TimeoutError:
                if self._bot_speaking:
                    await self._bot_stopped_speaking()
                continue
            if not self._bot_speaking:
                await self._bot_started_speaking()
            await self.write_audio(chunk)

    async def _bot_started_speaking(self):
        self._bot_speaking = True
        logger.info("Bot started speaking")
        await self.push_frame(BotStartedSpeakingFrame())
        await self.push_frame(BotStartedSpeakingFrame(), FrameDirection.UPSTREAM)

    async def _bot_stopped_speaking(self):
        self._bot_speaking = False
        logger.info("Bot stopped speaking")
        await self.push_frame(BotStoppedSpeakingFrame())
        await self.push_frame(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)


class BaseTransport:
    """A transport hands out one input and one output processor."""

    def input(self) -> BaseInputTransport:
        raise NotImplementedError

    def output(self) -> BaseOutputTransport:
        raise NotImplementedError
