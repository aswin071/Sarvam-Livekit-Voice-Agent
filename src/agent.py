import json
import logging
from pathlib import Path
from dotenv import load_dotenv
from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.voice import Agent, AgentSession
from livekit.plugins import anthropic, sarvam

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env.local")

logger = logging.getLogger("voice-agent")
logger.setLevel(logging.INFO)

# Memory file to persist user data across sessions
MEMORY_FILE = Path(__file__).parent / "memory.json"


def load_memory() -> dict:
    if MEMORY_FILE.exists():
        return json.loads(MEMORY_FILE.read_text())
    return {}


def save_memory(data: dict):
    MEMORY_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False))


class VoiceAgent(Agent):
    def __init__(self, user_memory: dict) -> None:
        # Build memory context to inject into prompt
        memory_context = ""
        if user_memory:
            memory_context = f"""
                Returning user memory:
                - Name: {user_memory.get('name', 'Unknown')}
                - Last loan interest: {user_memory.get('loan_type', 'Not asked')}
                - Last amount discussed: {user_memory.get('loan_amount', 'Not asked')}
                - Language preference: {user_memory.get('language', 'English')}

                Greet them by name and acknowledge their previous enquiry naturally.
            """

        super().__init__(
            instructions=f"""
                You are Arjun, a friendly loan enquiry agent at FinEase Bank. Speak like a real human.

                {memory_context}

                Language:
                - Detect what language the user speaks and always respond in the same language
                - Support all Indian languages: Hindi, Tamil, Telugu, Malayalam, Kannada, Bengali, Gujarati, Marathi, Punjabi, and English
                - Handle code-mixed speech like Hinglish or Tanglish naturally

                Conversation style:
                - Ask only one question at a time
                - Use short, natural sentences
                - Use natural acknowledgements in the user's language
                - Never use bullet points, lists, asterisks, or emojis
                - Always acknowledge what the user said before asking the next question

                Flow:
                1. Greet warmly and ask for their name (skip if returning user)
                2. Ask what type of loan they need (home, car, personal, business)
                3. Ask for loan amount and tenure
                4. Calculate and explain EMI clearly (use formula: EMI = P*r*(1+r)^n / ((1+r)^n - 1))
                5. Ask if they want to proceed or have more questions

                Memory instructions:
                - When you learn the user's name, say [MEMORY:name=VALUE]
                - When you learn the loan type, say [MEMORY:loan_type=VALUE]
                - When you learn the loan amount, say [MEMORY:loan_amount=VALUE]
                - When you detect their language, say [MEMORY:language=VALUE]
                - Put these tags at the very end of your response, they will be hidden from the user
            """,

            stt=sarvam.STT(
                language="unknown",
                model="saaras:v3",
                mode="transcribe",
                flush_signal=True
            ),

            llm=anthropic.LLM(model="claude-haiku-4-5-20251001"),

            tts=sarvam.TTS(
                target_language_code="unknown",
                model="bulbul:v3",
                speaker="rahul",
                speech_rate=1.05
            ),
        )
        self.user_memory = user_memory.copy()

    async def on_enter(self):
        self.session.generate_reply()

    async def on_user_turn_completed(self, turn_ctx, new_message):
        """Parse and save memory tags from LLM responses"""
        await super().on_user_turn_completed(turn_ctx, new_message)

    async def stt_node(self, audio, model_settings=None):
        """Process STT and extract memory from responses"""
        async for event in super().stt_node(audio, model_settings):
            yield event


async def entrypoint(ctx: JobContext):
    logger.info(f"User connected: {ctx.room.name}")

    # Load existing memory for this room/user
    all_memory = load_memory()
    user_memory = all_memory.get(ctx.room.name, {})

    if user_memory:
        logger.info(f"Returning user memory loaded: {user_memory}")
    else:
        logger.info("New user — no memory found")

    session = AgentSession(
        turn_detection="stt",
        min_endpointing_delay=0.07,
    )

    agent = VoiceAgent(user_memory=user_memory)
    await session.start(agent=agent, room=ctx.room)

    # Save memory when session ends
    @ctx.room.on("disconnected")
    def on_disconnected(*args):
        if agent.user_memory:
            all_memory[ctx.room.name] = agent.user_memory
            save_memory(all_memory)
            logger.info(f"Memory saved for {ctx.room.name}: {agent.user_memory}")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
