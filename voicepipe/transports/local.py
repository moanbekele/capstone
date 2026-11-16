"""Local transport: your computer's microphone and speakers (via sounddevice)."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import sounddevice as sd

from ..frames import StartFrame
from .base import BaseInputTransport, BaseOutputTransport, BaseTransport, TransportParams


class LocalAudioInputTransport(BaseInputTransport):
    def __init__(self, params: TransportParams):
        super().__init__(params, name="LocalAudioInput")
        self._stream: Optional[sd.RawInputStream] = None

    async def start(self, frame: StartFrame):
        await super().start(frame)
        loop = asyncio.get_running_loop()
        queue = self._audio_in_queue

        def on_audio(indata, frames, time_info, status):
            # Runs on sounddevice's thread: hand the bytes to the event loop.
            try:
                loop.call_soon_threadsafe(queue.put_nowait, bytes(indata))
            except RuntimeError:
                pass  # loop already closed while shutting down

        sample_rate = self._params.audio_in_sample_rate
        self._stream = sd.RawInputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
            blocksize=int(sample_rate * self._params.audio_in_chunk_ms / 1000),
            callback=on_audio,
        )
        self._stream.start()

    async def stop(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        await super().stop()


class LocalAudioOutputTransport(BaseOutputTransport):
    def __init__(self, params: TransportParams):
        super().__init__(params, name="LocalAudioOutput")
        self._stream: Optional[sd.RawOutputStream] = None
        self._executor = ThreadPoolExecutor(max_workers=1)

    async def start(self, frame: StartFrame):
        self._stream = sd.RawOutputStream(
            samplerate=self._params.audio_out_sample_rate,
            channels=1,
            dtype="int16",
        )
        self._stream.start()
        await super().start(frame)

    async def write_audio(self, audio: bytes):
        if self._stream is not None:
            # stream.write blocks until the device takes the audio, so run it off-loop.
            await asyncio.get_running_loop().run_in_executor(self._executor, self._stream.write, audio)

    async def stop(self):
        await super().stop()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


class LocalAudioTransport(BaseTransport):
    """Talk to the bot through this computer's default mic and speakers."""

    def __init__(self, params: TransportParams):
        self._params = params
        self._input: Optional[LocalAudioInputTransport] = None
        self._output: Optional[LocalAudioOutputTransport] = None

    def input(self) -> LocalAudioInputTransport:
        if self._input is None:
            self._input = LocalAudioInputTransport(self._params)
        return self._input

    def output(self) -> LocalAudioOutputTransport:
        if self._output is None:
            self._output = LocalAudioOutputTransport(self._params)
        return self._output
