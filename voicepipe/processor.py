"""FrameProcessor: the building block of every pipeline."""

from __future__ import annotations

import asyncio
import logging
from enum import Enum
from typing import Coroutine, List, Optional, Set

from .frames import ErrorFrame, Frame, StartFrame, StartInterruptionFrame, SystemFrame

logger = logging.getLogger(__name__)


class FrameDirection(Enum):
    DOWNSTREAM = 1  # towards the speaker
    UPSTREAM = 2  # back towards the microphone


class FrameProcessor:
    """Receives frames, does its job, and pushes frames to its neighbours.

    Every processor runs two asyncio tasks:

    * the input task receives every frame. SystemFrames are handled there right
      away; everything else is passed to the process queue.
    * the process task handles data and control frames one at a time, in order.

    An interruption cancels only the process task. Queued data (LLM tokens, TTS
    audio) is thrown away while system frames keep flowing.

    Subclasses override process_frame(), call super() first, and push on any
    frame they don't consume:

        async def process_frame(self, frame, direction):
            await super().process_frame(frame, direction)
            if isinstance(frame, TextFrame):
                await self.push_frame(TextFrame(text=frame.text.upper()))
            else:
                await self.push_frame(frame, direction)
    """

    def __init__(self, *, name: Optional[str] = None):
        self.name = name or type(self).__name__
        self._prev: Optional[FrameProcessor] = None
        self._next: Optional[FrameProcessor] = None
        self._allow_interruptions = False
        self._input_queue: Optional[asyncio.Queue] = None
        self._process_queue: Optional[asyncio.Queue] = None
        self._input_task: Optional[asyncio.Task] = None
        self._process_task: Optional[asyncio.Task] = None
        self._tasks: Set[asyncio.Task] = set()

    def __str__(self):
        return self.name

    # --- wiring --------------------------------------------------------------

    def link(self, processor: "FrameProcessor") -> "FrameProcessor":
        """Connect self -> processor. Returns processor so links can chain."""
        self._next = processor
        processor._prev = self
        return processor

    @property
    def processors(self) -> List["FrameProcessor"]:
        """Child processors. Only containers such as Pipeline have any."""
        return []

    @property
    def interruptions_allowed(self) -> bool:
        return self._allow_interruptions

    # --- lifecycle -----------------------------------------------------------

    async def setup(self):
        """Create the input and process tasks. Called by PipelineTask."""
        self._input_queue = asyncio.Queue()
        self._input_task = asyncio.create_task(self._input_loop(), name=f"{self}::input")
        self._create_process_task()

    async def cleanup(self):
        """Cancel every task this processor owns. Called by PipelineTask."""
        for task in list(self._tasks):
            await self.cancel_task(task)
        await self.cancel_task(self._process_task)
        await self.cancel_task(self._input_task)
        self._process_task = self._input_task = None

    def create_task(self, coro: Coroutine, name: Optional[str] = None) -> asyncio.Task:
        """Start a background task that is cancelled on cleanup()."""
        task = asyncio.create_task(coro, name=name or f"{self}::task")
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    @staticmethod
    async def cancel_task(task: Optional[asyncio.Task], timeout: float = 1.0):
        if task is None or task.done() or task is asyncio.current_task():
            return
        task.cancel()
        await asyncio.wait({task}, timeout=timeout)

    # --- receiving frames ----------------------------------------------------

    async def queue_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        """Hand a frame to this processor. Called by its neighbours."""
        if self._input_queue is None:
            raise RuntimeError(f"{self} got {frame} before setup()")
        await self._input_queue.put((frame, direction))

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Override this. The base class only tracks pipeline settings."""
        if isinstance(frame, StartFrame):
            self._allow_interruptions = frame.allow_interruptions

    async def _input_loop(self):
        while True:
            frame, direction = await self._input_queue.get()
            if isinstance(frame, SystemFrame):
                if isinstance(frame, StartInterruptionFrame):
                    await self._start_interruption()
                await self._call_process_frame(frame, direction)
            else:
                await self._process_queue.put((frame, direction))

    async def _process_loop(self):
        while True:
            frame, direction = await self._process_queue.get()
            await self._call_process_frame(frame, direction)

    async def _call_process_frame(self, frame: Frame, direction: FrameDirection):
        try:
            await self.process_frame(frame, direction)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception("%s failed while processing %s", self, frame)
            await self.push_error(f"{self}: {e}")

    def _create_process_task(self):
        self._process_queue = asyncio.Queue()
        self._process_task = asyncio.create_task(self._process_loop(), name=f"{self}::process")

    async def _start_interruption(self):
        """Throw away queued work and whatever is being processed now."""
        await self.cancel_task(self._process_task)
        self._create_process_task()

    # --- sending frames ------------------------------------------------------

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        target = self._next if direction == FrameDirection.DOWNSTREAM else self._prev
        if target is not None:
            await target.queue_frame(frame, direction)

    async def push_error(self, error: str, fatal: bool = False):
        await self.push_frame(ErrorFrame(error=error, fatal=fatal), FrameDirection.UPSTREAM)
