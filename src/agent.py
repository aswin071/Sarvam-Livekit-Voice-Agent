# import logging

# from dotenv import load_dotenv
# from livekit import rtc
# from livekit.agents import (
#     Agent,
#     AgentServer,
#     AgentSession,
#     JobContext,
#     JobProcess,
#     cli,
#     inference,
#     room_io,
# )
# from livekit.plugins import noise_cancellation, silero
# from livekit.plugins.turn_detector.multilingual import MultilingualModel

# logger = logging.getLogger("agent")

# load_dotenv(".env.local")


# class Assistant(Agent):
#     def __init__(self) -> None:
#         super().__init__(
#             instructions="""You are a helpful voice AI assistant. The user is interacting with you via voice, even if you perceive the conversation as text.
#             You eagerly assist users with their questions by providing information from your extensive knowledge.
#             Your responses are concise, to the point, and without any complex formatting or punctuation including emojis, asterisks, or other symbols.
#             You are curious, friendly, and have a sense of humor.""",
#         )

#     # To add tools, use the @function_tool decorator.
#     # Here's an example that adds a simple weather tool.
#     # You also have to add `from livekit.agents import function_tool, RunContext` to the top of this file
#     # @function_tool
#     # async def lookup_weather(self, context: RunContext, location: str):
#     #     """Use this tool to look up current weather information in the given location.
#     #
#     #     If the location is not supported by the weather service, the tool will indicate this. You must tell the user the location's weather is unavailable.
#     #
#     #     Args:
#     #         location: The location to look up weather information for (e.g. city name)
#     #     """
#     #
#     #     logger.info(f"Looking up weather for {location}")
#     #
#     #     return "sunny with a temperature of 70 degrees."


# server = AgentServer()


# def prewarm(proc: JobProcess):
#     proc.userdata["vad"] = silero.VAD.load()


# server.setup_fnc = prewarm


# @server.rtc_session(agent_name="my-agent")
# async def my_agent(ctx: JobContext):
#     # Logging setup
#     # Add any other context you want in all log entries here
#     ctx.log_context_fields = {
#         "room": ctx.room.name,
#     }

#     # Set up a voice AI pipeline using OpenAI, Cartesia, Deepgram, and the LiveKit turn detector
#     session = AgentSession(
#         # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
#         # See all available models at https://docs.livekit.io/agents/models/stt/
#         stt=inference.STT(model="deepgram/nova-3", language="multi"),
#         # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
#         # See all available models at https://docs.livekit.io/agents/models/llm/
#         llm=inference.LLM(model="openai/gpt-4.1-mini"),
#         # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
#         # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/
#         tts=inference.TTS(
#             model="cartesia/sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"
#         ),
#         # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
#         # See more at https://docs.livekit.io/agents/build/turns
#         turn_detection=MultilingualModel(),
#         vad=ctx.proc.userdata["vad"],
#         # allow the LLM to generate a response while waiting for the end of turn
#         # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
#         preemptive_generation=True,
#     )

#     # To use a realtime model instead of a voice pipeline, use the following session setup instead.
#     # (Note: This is for the OpenAI Realtime API. For other providers, see https://docs.livekit.io/agents/models/realtime/))
#     # 1. Install livekit-agents[openai]
#     # 2. Set OPENAI_API_KEY in .env.local
#     # 3. Add `from livekit.plugins import openai` to the top of this file
#     # 4. Use the following session setup instead of the version above
#     # session = AgentSession(
#     #     llm=openai.realtime.RealtimeModel(voice="marin")
#     # )

#     # # Add a virtual avatar to the session, if desired
#     # # For other providers, see https://docs.livekit.io/agents/models/avatar/
#     # avatar = hedra.AvatarSession(
#     #   avatar_id="...",  # See https://docs.livekit.io/agents/models/avatar/plugins/hedra
#     # )
#     # # Start the avatar and wait for it to join
#     # await avatar.start(session, room=ctx.room)

#     # Start the session, which initializes the voice pipeline and warms up the models
#     await session.start(
#         agent=Assistant(),
#         room=ctx.room,
#         room_options=room_io.RoomOptions(
#             audio_input=room_io.AudioInputOptions(
#                 noise_cancellation=lambda params: (
#                     noise_cancellation.BVCTelephony()
#                     if params.participant.kind
#                     == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
#                     else noise_cancellation.BVC()
#                 ),
#             ),
#         ),
#     )

#     # Join the room and connect to the user
#     await ctx.connect()


# if __name__ == "__main__":
#     cli.run_app(server)



import logging
from pathlib import Path
from dotenv import load_dotenv
from livekit.agents import JobContext, WorkerOptions, cli

from livekit.agents.voice import Agent, AgentSession
from livekit.plugins import anthropic, sarvam

# Load environment variables from .env.local in the parent directory
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env.local")

# Set up logging
logger = logging.getLogger("voice-agent")
logger.setLevel(logging.INFO)


class VoiceAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            # Your agent's personality and instructions
            instructions="""
                You are Arjun, a friendly loan enquiry agent. Speak like a real human, not a robot.

                Language:
                - Detect what language the user is speaking and always respond in the same language
                - You understand and speak all Indian languages: Hindi, Tamil, Telugu, Malayalam, Kannada, Bengali, Gujarati, Marathi, Punjabi, Odia, and English
                - Also handle code-mixed speech like Hinglish or Tanglish naturally

                Conversation style:
                - Ask only one question at a time
                - Use short, natural sentences
                - Use natural acknowledgements like "Got it", "Sure", "Of course" in the user's language
                - Never use bullet points, lists, asterisks, or emojis
                - Always acknowledge what the user said before asking the next question

                Flow:
                1. Greet warmly in English first, then switch to user's language
                2. Ask what type of loan they need (home, car, personal)
                3. Ask for loan amount and tenure
                4. Explain the EMI clearly and simply
            """,

            # Saaras v3 STT - Auto-detects all Indian languages
            stt=sarvam.STT(
                language="unknown",
                model="saaras:v3",
                mode="transcribe",
                flush_signal=True
            ),

            # Claude LLM - Haiku for low-latency voice
            llm=anthropic.LLM(model="claude-haiku-4-5-20251001"),

            # Bulbul TTS - Auto language detection
            tts=sarvam.TTS(
                target_language_code="unknown",
                model="bulbul:v3",
                speaker="rahul",
                speech_rate=1.1
            ),
        )
    
    async def on_enter(self):
        """Called when user joins - agent starts the conversation"""
        self.session.generate_reply()


async def entrypoint(ctx: JobContext):
    """Main entry point - LiveKit calls this when a user connects"""
    logger.info(f"User connected to room: {ctx.room.name}")

    session = AgentSession(
        turn_detection="stt",
        min_endpointing_delay=0.07,
    )
    await session.start(
        agent=VoiceAgent(),
        room=ctx.room
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
