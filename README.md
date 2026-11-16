# voicepipe

A small Python framework for real-time voice agents, modeled on
[Pipecat](https://github.com/pipecat-ai/pipecat). The example bot runs on
Groq's free API: Whisper turns speech into text, `gpt-oss` (OpenAI's open
model) writes the replies, and Orpheus speaks them. OpenAI versions of the
same services are in `voicepipe/services/openai_service.py`.

```
mic ─► VAD ─► speech-to-text ─► LLM ─► text-to-speech ─► speaker
```

## Run it

1. Install the dependencies: `pip install -r requirements.txt`
2. Copy `.env.example` to `.env` and put your Groq API key in it
   (free at console.groq.com/keys).
3. Accept the Orpheus voice model's terms once, at
   console.groq.com/playground?model=canopylabs/orpheus-v1-english
4. Run `python bot.py`, then talk. Press Ctrl+C to quit.

Wear headphones. With speakers, the mic picks up the bot's voice and the bot
keeps interrupting itself.

## How it works

The same ideas as Pipecat, in about 1,000 lines:

| Piece | File | What it does |
|---|---|---|
| Frames | `voicepipe/frames.py` | Everything that moves through the pipeline: audio, text, start/end, interruptions. |
| FrameProcessor | `voicepipe/processor.py` | One step in the pipeline. Receives frames, pushes frames to its neighbours. |
| Pipeline / PipelineTask / PipelineRunner | `voicepipe/pipeline.py` | Link the processors in order and run them. |
| VAD | `voicepipe/vad.py` | Works out when the user starts and stops talking (loudness-based). |
| Transports | `voicepipe/transports/` | Where audio comes in and goes out. `LocalAudioTransport` uses your mic and speakers. |
| Context aggregators | `voicepipe/llm_context.py` | Keep the conversation history: what the user said and what the bot said. |
| Services | `voicepipe/services/` | Base classes for STT/LLM/TTS, plus the OpenAI versions. |

**Frame types.** `SystemFrame`s (mic audio, start, interruptions) are handled
right away. `DataFrame`s and `ControlFrame`s (text, bot audio, end of reply)
are handled in order. Every processor runs two asyncio tasks, one for each.

**Interruptions.** When the user starts talking, the input transport sends a
`StartInterruptionFrame`. Each processor cancels its processing task, which
drops the rest of the ChatGPT reply, any queued speech and any unplayed audio.
The part of the reply already spoken stays in the conversation history.

**Low latency.** ChatGPT's reply is streamed token by token. The TTS service
speaks each sentence as soon as it's complete, so the bot starts talking
before the whole reply has been written.

## Writing your own processor

```python
from voicepipe.frames import TextFrame
from voicepipe.processor import FrameProcessor

class Shouter(FrameProcessor):
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, TextFrame):
            frame.text = frame.text.upper()
        await self.push_frame(frame, direction)
```

Put it anywhere in the list passed to `Pipeline([...])`.
