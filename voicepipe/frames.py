"""Frame types. Everything that moves through a pipeline is a Frame.

Three families, which decide how a processor handles them:

* SystemFrame  - handled immediately, ahead of queued work, and never dropped
                 by an interruption (audio input, start/cancel, interruptions).
* DataFrame    - content (text, audio, LLM context). Handled in order and
                 dropped when the user interrupts the bot.
* ControlFrame - in-band signals (end of stream, LLM response boundaries).
                 Handled in order together with data frames.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, List

_frame_ids = itertools.count(1)


@dataclass
class Frame:
    id: int = field(init=False)
    name: str = field(init=False)

    def __post_init__(self):
        self.id = next(_frame_ids)
        self.name = f"{type(self).__name__}#{self.id}"

    def __str__(self):
        return self.name


@dataclass
class SystemFrame(Frame):
    pass


@dataclass
class DataFrame(Frame):
    pass


@dataclass
class ControlFrame(Frame):
    pass


# --- System frames -----------------------------------------------------------


@dataclass
class StartFrame(SystemFrame):
    """First frame through the pipeline. Carries pipeline-wide settings."""

    allow_interruptions: bool = True
    audio_in_sample_rate: int = 16000
    audio_out_sample_rate: int = 24000


@dataclass
class CancelFrame(SystemFrame):
    """Stop now, without waiting for queued work."""


@dataclass
class ErrorFrame(SystemFrame):
    """Pushed upstream when a processor fails. Fatal errors stop the task."""

    error: str
    fatal: bool = False


@dataclass
class EndTaskFrame(SystemFrame):
    """Pushed upstream by a processor to ask the task to end gracefully."""


@dataclass
class StartInterruptionFrame(SystemFrame):
    """The user started talking over the bot: drop queued output."""


@dataclass
class StopInterruptionFrame(SystemFrame):
    """The user stopped talking after an interruption."""


@dataclass
class UserStartedSpeakingFrame(SystemFrame):
    pass


@dataclass
class UserStoppedSpeakingFrame(SystemFrame):
    pass


@dataclass
class BotStartedSpeakingFrame(SystemFrame):
    pass


@dataclass
class BotStoppedSpeakingFrame(SystemFrame):
    pass


@dataclass
class InputAudioRawFrame(SystemFrame):
    """A chunk of 16-bit PCM audio from the microphone."""

    audio: bytes
    sample_rate: int
    num_channels: int = 1


# --- Data frames -------------------------------------------------------------


@dataclass
class OutputAudioRawFrame(DataFrame):
    """A chunk of 16-bit PCM audio to play on the speaker."""

    audio: bytes
    sample_rate: int
    num_channels: int = 1


@dataclass
class TTSAudioRawFrame(OutputAudioRawFrame):
    """Audio produced by a text-to-speech service."""


@dataclass
class TextFrame(DataFrame):
    text: str


@dataclass
class LLMTextFrame(TextFrame):
    """A token (or a few) streamed from the LLM."""


@dataclass
class TTSTextFrame(TextFrame):
    """Text that the TTS service has turned into audio."""


@dataclass
class TranscriptionFrame(TextFrame):
    """What the user said, from the speech-to-text service."""

    user_id: str = "user"
    timestamp: str = ""


@dataclass
class TTSSpeakFrame(DataFrame):
    """Ask the TTS service to say this text directly, bypassing the LLM."""

    text: str


@dataclass
class LLMMessagesFrame(DataFrame):
    """Replace the conversation with these messages and run the LLM."""

    messages: List[dict]


@dataclass
class LLMContextFrame(DataFrame):
    """Run the LLM on this conversation context."""

    context: Any


# --- Control frames ----------------------------------------------------------


@dataclass
class EndFrame(ControlFrame):
    """Finish whatever is queued, then shut the pipeline down."""


@dataclass
class LLMFullResponseStartFrame(ControlFrame):
    pass


@dataclass
class LLMFullResponseEndFrame(ControlFrame):
    pass


@dataclass
class TTSStartedFrame(ControlFrame):
    pass


@dataclass
class TTSStoppedFrame(ControlFrame):
    pass
