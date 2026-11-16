"""voicepipe: a small Pipecat-style framework for real-time voice agents.

Audio flows through a Pipeline of FrameProcessors:

    mic -> VAD -> speech-to-text -> LLM -> text-to-speech -> speaker

Each step is a FrameProcessor; everything that moves between them is a Frame.
"""

__version__ = "0.1.0"
