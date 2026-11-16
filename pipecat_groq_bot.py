"""Groq voice agent with Pipecat's web UI.

Runs on the Pipecat source in ./pipecat. Whisper hears you, gpt-oss writes the
reply and Orpheus speaks it, all on Groq. The browser connects over WebRTC
through Pipecat's prebuilt UI.

    uv venv .venv --python 3.13
    uv pip install --python .venv\\Scripts\\python.exe -e "./pipecat[groq,webrtc,runner]" python-dotenv
    .venv\\Scripts\\python.exe pipecat_groq_bot.py

Then open http://localhost:7860/client and press Connect.
"""

import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker, ProcessorUnusablePolicy
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.groq.llm import GroqLLMService
from pipecat.services.groq.stt import GroqSTTService
from pipecat.services.groq.tts import GroqTTSService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)

SYSTEM_PROMPT = (
    "You are a friendly voice assistant. Everything you write is spoken aloud, "
    "so keep replies to one to three short sentences and never use lists, "
    "markdown, emojis or special characters."
)

# Only the browser (WebRTC) transport is used, so no Daily/LiveKit/Twilio packages are needed.
transport_params = {
    "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    api_key = os.environ["GROQ_API_KEY"]

    stt = GroqSTTService(api_key=api_key)
    llm = GroqLLMService(
        api_key=api_key,
        settings=GroqLLMService.Settings(
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
            system_instruction=SYSTEM_PROMPT,
        ),
    )
    tts = GroqTTSService(
        api_key=api_key,
        settings=GroqTTSService.Settings(voice=os.getenv("GROQ_VOICE", "troy")),
    )

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline(
        [
            transport.input(),  # audio from the browser
            stt,  # speech -> text (Whisper)
            user_aggregator,  # add what the user said to the conversation
            llm,  # write the reply (gpt-oss)
            tts,  # text -> speech (Orpheus)
            transport.output(),  # audio back to the browser
            assistant_aggregator,  # add what the bot said to the conversation
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
        processor_unusable_policy=ProcessorUnusablePolicy.END,
    )
    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        context.add_message({"role": "developer", "content": "Greet the user and introduce yourself briefly."})
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await runner.cancel()

    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Entry point the Pipecat runner calls for each browser session."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
