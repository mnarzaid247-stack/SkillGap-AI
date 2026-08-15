import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from agents.career_coach import CareerCoachAgent
from agents.supervisor import SupervisorAgent


load_dotenv()


def create_llm() -> ChatOpenAI:
    api_key = os.getenv(
        "OPENROUTER_API_KEY"
    )

    model_name = os.getenv(
        "OPENROUTER_MODEL"
    )

    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY is missing"
        )

    if not model_name:
        raise ValueError(
            "OPENROUTER_MODEL is missing"
        )

    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.1,
    )


llm = create_llm()
career_coach_agent = CareerCoachAgent()
supervisor_agent = SupervisorAgent()


def _live_log(message: str) -> None:
    """Print workflow progress immediately in the terminal."""
    print(message, flush=True)
