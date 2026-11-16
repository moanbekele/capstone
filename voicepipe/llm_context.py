"""Conversation context and the aggregators that keep it up to date."""

from __future__ import annotations

import logging
from typing import List, Optional

from .frames import (
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesFrame,
    StartInterruptionFrame,
    TextFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from .processor import FrameProcessor

logger = logging.getLogger(__name__)


class LLMContext:
    """The conversation so far, as OpenAI-style chat messages."""

    def __init__(self, messages: Optional[List[dict]] = None):
        self.messages: List[dict] = list(messages or [])

    def add_message(self, message: dict):
        self.messages.append(message)

    def set_messages(self, messages: List[dict]):
        self.messages = list(messages)

    def get_messages(self) -> List[dict]:
        return list(self.messages)


class LLMUserContextAggregator(FrameProcessor):
    """Collects what the user said and asks the LLM to reply once their turn ends."""

    def __init__(self, context: LLMContext, **kwargs):
        super().__init__(**kwargs)
        self._context = context
        self._aggregation: List[str] = []
        self._user_speaking = False

    def get_context_frame(self) -> LLMContextFrame:
        """A frame that makes the LLM respond to the current context."""
        return LLMContextFrame(context=self._context)

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, UserStartedSpeakingFrame):
            self._user_speaking = True
            await self.push_frame(frame, direction)
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._user_speaking = False
            await self.push_frame(frame, direction)
            await self._push_aggregation()
        elif isinstance(frame, TranscriptionFrame):
            if frame.text.strip():
                self._aggregation.append(frame.text.strip())
            # Transcripts usually arrive just after the user stops talking.
            if not self._user_speaking:
                await self._push_aggregation()
        elif isinstance(frame, LLMMessagesFrame):
            self._context.set_messages(frame.messages)
            await self.push_frame(self.get_context_frame())
        else:
            await self.push_frame(frame, direction)

    async def _push_aggregation(self):
        if not self._aggregation:
            return
        text = " ".join(self._aggregation)
        self._aggregation = []
        self._context.add_message({"role": "user", "content": text})
        await self.push_frame(self.get_context_frame())


class LLMAssistantContextAggregator(FrameProcessor):
    """Records what the bot said. Put it after the output transport."""

    def __init__(self, context: LLMContext, **kwargs):
        super().__init__(**kwargs)
        self._context = context
        self._aggregation: List[str] = []

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMFullResponseStartFrame):
            self._aggregation = []
            await self.push_frame(frame, direction)
        elif isinstance(frame, LLMFullResponseEndFrame):
            await self.push_frame(frame, direction)
            self._commit()
        elif isinstance(frame, StartInterruptionFrame):
            self._commit()  # keep what was said before the user cut in
            await self.push_frame(frame, direction)
        elif isinstance(frame, TextFrame) and not isinstance(frame, TranscriptionFrame):
            self._aggregation.append(frame.text)
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)

    def _commit(self):
        text = " ".join(t.strip() for t in self._aggregation if t.strip())
        self._aggregation = []
        if text:
            self._context.add_message({"role": "assistant", "content": text})


class LLMContextAggregatorPair:
    """The user and assistant aggregators that share one context."""

    def __init__(self, context: LLMContext):
        self._user = LLMUserContextAggregator(context, name="UserAggregator")
        self._assistant = LLMAssistantContextAggregator(context, name="AssistantAggregator")

    def user(self) -> LLMUserContextAggregator:
        return self._user

    def assistant(self) -> LLMAssistantContextAggregator:
        return self._assistant
