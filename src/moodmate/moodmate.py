from agents import (
    Agent,
    Runner,
    AsyncOpenAI,
    set_default_openai_client,
    set_default_openai_api,
    set_tracing_disabled,
    OpenAIChatCompletionsModel,
)
from agents.run import RunConfig
from dotenv import load_dotenv
import os
from rich import print

load_dotenv()


gemini_api_key = os.getenv("GEMINI_API_KEY")

external_provider = AsyncOpenAI(
    api_key=gemini_api_key, base_url="https://generativelanguage.googleapis.com/v1beta/"
)


model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash", openai_client=external_provider
)


def moodmate():
    moodmate = Agent(
        name="MoodMate",
        instructions="""
    You are MoodMate, a friendly and emotionally intelligent assistant.
    Your job is to help users reflect on their emotions and offer general tips for self-care.
    Be supportive, non-judgmental, and never offer medical advice.
    Use soft, comforting language and speak like a calm friend or life coach.
    Always end your response with a gentle question to encourage continued reflection.
    """,
    )

    run_config = RunConfig(
        model=model,
        tracing_disabled=True,
    )

    result = Runner.run_sync(
        moodmate, "I've been feeling really anxious and low energy lately.", run_config=run_config
    )

    print(result.final_output)

    with open("D:\doc data\moodmate\output.md", "w", encoding="utf-8") as file:
        file.write(result.final_output)

    print("✅ output.md updated!")
