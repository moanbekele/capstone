"""Pipeline, PipelineTask and PipelineRunner."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Awaitable, Callable, Iterable, Iterator, List, Optional

from .frames import CancelFrame, EndFrame, EndTaskFrame, ErrorFrame, Frame, StartFrame
from .processor import FrameDirection, FrameProcessor

logger = logging.getLogger(__name__)

PushFn = Callable[[Frame, FrameDirection], Awaitable[None]]


class _PipelineSource(FrameProcessor):
    """First link inside a Pipeline. Upstream frames leave the pipeline here."""

    def __init__(self, upstream_push: PushFn):
        super().__init__(name="PipelineSource")
        self._upstream_push = upstream_push

    async def queue_frame(self, frame, direction=FrameDirection.DOWNSTREAM):
        if direction == FrameDirection.DOWNSTREAM:
            await self.push_frame(frame, direction)
        else:
            await self._upstream_push(frame, direction)


class _PipelineSink(FrameProcessor):
    """Last link inside a Pipeline. Downstream frames leave the pipeline here."""

    def __init__(self, downstream_push: PushFn):
        super().__init__(name="PipelineSink")
        self._downstream_push = downstream_push

    async def queue_frame(self, frame, direction=FrameDirection.DOWNSTREAM):
        if direction == FrameDirection.DOWNSTREAM:
            await self._downstream_push(frame, direction)
        else:
            await self.push_frame(frame, direction)


class Pipeline(FrameProcessor):
    """Links processors in order. A Pipeline is itself a processor, so it can nest."""

    def __init__(self, processors: Iterable[FrameProcessor], *, name: Optional[str] = None):
        super().__init__(name=name)
        self._processors: List[FrameProcessor] = list(processors)
        self._source = _PipelineSource(self.push_frame)
        self._sink = _PipelineSink(self.push_frame)
        current: FrameProcessor = self._source
        for processor in self._processors:
            current = current.link(processor)
        current.link(self._sink)

    @property
    def processors(self) -> List[FrameProcessor]:
        return self._processors

    async def setup(self):
        pass  # A pipeline only routes frames; its children do the work.

    async def queue_frame(self, frame, direction=FrameDirection.DOWNSTREAM):
        if direction == FrameDirection.DOWNSTREAM:
            await self._source.queue_frame(frame, direction)
        else:
            await self._sink.queue_frame(frame, direction)


@dataclass
class PipelineParams:
    allow_interruptions: bool = True
    audio_in_sample_rate: int = 16000
    audio_out_sample_rate: int = 24000


class _TaskSource(FrameProcessor):
    def __init__(self, task: "PipelineTask"):
        super().__init__(name="TaskSource")
        self._task = task

    async def queue_frame(self, frame, direction=FrameDirection.DOWNSTREAM):
        if direction == FrameDirection.UPSTREAM:
            await self._task._handle_upstream(frame)
        else:
            await self.push_frame(frame, direction)


class _TaskSink(FrameProcessor):
    def __init__(self, task: "PipelineTask"):
        super().__init__(name="TaskSink")
        self._task = task

    async def queue_frame(self, frame, direction=FrameDirection.DOWNSTREAM):
        if isinstance(frame, (EndFrame, CancelFrame)):
            self._task._finished.set()


class PipelineTask:
    """Runs one pipeline: sets up processors, sends StartFrame, and waits for the end."""

    def __init__(self, pipeline: FrameProcessor, params: Optional[PipelineParams] = None):
        self.pipeline = pipeline
        self.params = params or PipelineParams()
        self._source = _TaskSource(self)
        self._sink = _TaskSink(self)
        self._source.link(pipeline)
        pipeline.link(self._sink)
        self._pending: asyncio.Queue = asyncio.Queue()
        self._finished = asyncio.Event()

    async def queue_frame(self, frame: Frame):
        """Send a frame into the pipeline. Safe to call before run()."""
        await self._pending.put(frame)

    async def queue_frames(self, frames: Iterable[Frame]):
        for frame in frames:
            await self.queue_frame(frame)

    async def stop_when_done(self):
        """Finish what is queued (e.g. the bot's last sentence), then stop."""
        await self.queue_frame(EndFrame())

    async def cancel(self):
        """Stop right away."""
        if self._finished.is_set():
            return
        await self._source.push_frame(CancelFrame())
        try:
            await asyncio.wait_for(self._finished.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            self._finished.set()

    async def run(self):
        processors = list(self._walk(self.pipeline))
        for processor in processors:
            await processor.setup()
        start = StartFrame(
            allow_interruptions=self.params.allow_interruptions,
            audio_in_sample_rate=self.params.audio_in_sample_rate,
            audio_out_sample_rate=self.params.audio_out_sample_rate,
        )
        await self._source.push_frame(start)
        pump = asyncio.create_task(self._pump_pending(), name="PipelineTask::pump")
        try:
            await self._finished.wait()
        finally:
            pump.cancel()
            for processor in reversed(processors):
                await processor.cleanup()

    async def _pump_pending(self):
        while True:
            frame = await self._pending.get()
            await self._source.push_frame(frame)

    async def _handle_upstream(self, frame: Frame):
        if isinstance(frame, ErrorFrame):
            logger.error("Pipeline error: %s", frame.error)
            if frame.fatal:
                asyncio.create_task(self.cancel())
        elif isinstance(frame, EndTaskFrame):
            await self.queue_frame(EndFrame())

    def _walk(self, processor: FrameProcessor) -> Iterator[FrameProcessor]:
        yield processor
        for child in processor.processors:
            yield from self._walk(child)


class PipelineRunner:
    """Runs a PipelineTask until it ends or the program is interrupted (Ctrl+C)."""

    async def run(self, task: PipelineTask):
        logger.info("Pipeline running. Press Ctrl+C to stop.")
        try:
            await task.run()
        except asyncio.CancelledError:
            logger.info("Pipeline stopped.")
            raise
        logger.info("Pipeline finished.")
