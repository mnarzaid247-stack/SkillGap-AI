import os
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI


load_dotenv()


class PriorityGap(BaseModel):
    skill: str
    priority: Literal["HIGH", "MEDIUM", "LOW"]
    reason: str


class PortfolioProject(BaseModel):
    title: str
    description: str
    technologies: list[str]


class CareerCoachOutput(BaseModel):
    priority_gaps: list[PriorityGap] = Field(
        description="Top priority missing skills from the verified gap analysis."
    )
    learning_order: list[str]
    portfolio_project: PortfolioProject
    next_action: str
    apply_recommendation: str


class CareerCoachAgent:
    def __init__(self):
        api_key = os.getenv("OPENROUTER_API_KEY")
        model_name = os.getenv("OPENROUTER_MODEL")

        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is missing")

        if not model_name:
            raise ValueError("OPENROUTER_MODEL is missing")

        self.llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0.2,
        )

        self.structured_llm = self.llm.with_structured_output(
            CareerCoachOutput
        )

    def generate_recommendations(self, state: dict) -> CareerCoachOutput:
        prompt = self._build_prompt(state)

        result = self.structured_llm.invoke(prompt)

        return result

    def _build_prompt(self, state: dict) -> str:
        target_role = state.get("target_role", "")
        matched_skills = state.get("matched_required_skills", [])
        missing_skills = state.get("missing_required_skills", [])
        evidence_gaps = state.get("evidence_gaps", [])
        skill_coverage = state.get("skill_coverage", 0)
        supervisor_feedback = state.get("supervisor_feedback", "")

        return f"""
You are the Career Coach Agent for SkillGap AI.

Your responsibility is to convert VERIFIED gap-analysis results
into practical career recommendations.

TARGET ROLE:
{target_role}

MATCHED REQUIRED SKILLS:
{matched_skills}

MISSING REQUIRED SKILLS:
{missing_skills}

EVIDENCE GAPS:
{evidence_gaps}

SKILL COVERAGE:
{skill_coverage}%

SUPERVISOR FEEDBACK:
{supervisor_feedback}

RULES:

1. Do NOT re-analyze the CV.
2. Do NOT re-analyze the job advertisement.
3. Do NOT invent missing skills.
4. Priority gaps must come from MISSING REQUIRED SKILLS.
5. Return at most 3 priority gaps.
6. HIGH priority should be used for important required skills.
7. Recommend a realistic learning order.
8. Suggest ONE practical portfolio project.
9. The portfolio project should help demonstrate missing skills.
10. Do not guarantee employment or job acceptance.
11. Give a practical next action.
12. If supervisor feedback exists, improve the recommendation
    according to that feedback.
"""