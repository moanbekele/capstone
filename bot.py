"""Voice agent on Groq: talk to it with your microphone and speakers.

    pip install -r requirements.txt
    put GROQ_API_KEY in a .env file (free key from console.groq.com)
    python bot.py

Whisper hears you, gpt-oss (OpenAI's open model) writes the reply, and
Orpheus speaks it. Use headphones, otherwise the mic hears the bot and it
interrupts itself.
"""

import asyncio
import logging
import os
import sys

from dotenv import load_dotenv

from voicepipe.llm_context import LLMContext, LLMContextAggregatorPair
from voicepipe.pipeline import Pipeline, PipelineParams, PipelineRunner, PipelineTask
from voicepipe.services.groq_service import GroqLLMService, GroqSTTService, GroqTTSService
from voicepipe.transports.base import TransportParams
from voicepipe.transports.local import LocalAudioTransport
from voicepipe.vad import EnergyVADAnalyzer

SYSTEM_PROMPT = (
    "You are a friendly voice assistant. Everything you write is spoken aloud, "
    "so keep replies to one to three short sentences, and never use lists, "
    "markdown, emojis or special characters."
)


async def main():
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        sys.exit("GROQ_API_KEY is not set. Copy .env.example to .env and add your key.")

    transport = LocalAudioTransport(
        TransportParams(
            audio_in_sample_rate=16000,
            audio_out_sample_rate=24000,
            vad_analyzer=EnergyVADAnalyzer(),
        )
    )

    stt = GroqSTTService(language="en")
    llm = GroqLLMService(model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"))
    tts = GroqTTSService(voice=os.getenv("GROQ_VOICE", "troy"), direction="[friendly]")

    context = LLMContext([{"role": "system", "content": SYSTEM_PROMPT}])
    context_aggregator = LLMContextAggregatorPair(context)

    pipeline = Pipeline(
        [
            transport.input(),  # microphone audio + turn detection
            stt,  # speech -> text
            context_aggregator.user(),  # add what the user said to the conversation
            llm,  # the LLM writes the reply
            tts,  # text -> speech
            transport.output(),  # play it on the speakers
            context_aggregator.assistant(),  # add what the bot said to the conversation
        ]
    )

    task = PipelineTask(
        pipeline,
        PipelineParams(
            allow_interruptions=True,
            audio_in_sample_rate=16000,
            audio_out_sample_rate=24000,
        ),
    )

    # Have the bot speak first.
    context.add_message({"role": "system", "content": "Greet the user and introduce yourself briefly."})
    await task.queue_frames([context_aggregator.user().get_context_frame()])

    await PipelineRunner().run(task)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBye!")
