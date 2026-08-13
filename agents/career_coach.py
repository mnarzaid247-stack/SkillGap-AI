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

        gap_analysis = state.get("gap_analysis", {})

        matched_skills = gap_analysis.get(
            "matching_required_skills",
            []
        )

        missing_skills = gap_analysis.get(
            "missing_required_skills",
            []
        )

        evidence_gaps = gap_analysis.get(
            "evidence_gaps",
            []
        )

        skill_coverage = gap_analysis.get(
            "skill_coverage",
            0
        )

        existing_priority_gaps = gap_analysis.get(
            "priority_gaps",
            []
        )

        supervisor_feedback = state.get(
            "supervisor_feedback",
            ""
        )

        return f"""
You are the Career Coach Agent for SkillGap AI.

Your responsibility is to convert VERIFIED gap-analysis results
into practical and realistic career recommendations.

You must use only the information provided in the verified
gap analysis.

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

EXISTING PRIORITY GAPS:
{existing_priority_gaps}

SUPERVISOR FEEDBACK:
{supervisor_feedback}

RULES:

1. Do NOT re-analyze the CV.

2. Do NOT re-analyze the job advertisement.

3. Do NOT invent new missing skills.

4. Priority gaps must come only from
   MISSING REQUIRED SKILLS.

5. Return at most 3 priority gaps.

6. Use the existing verified priority gaps when useful,
   but do not introduce skills that are not present
   in MISSING REQUIRED SKILLS.

7. HIGH priority should be used for important required
   skills that directly affect readiness for the target role.

8. Recommend a realistic learning order based on the
   verified missing skills.

9. Suggest exactly ONE practical portfolio project.

10. The portfolio project should help demonstrate one
    or more verified missing skills.

11. The portfolio project may also use already matched
    skills when useful.

12. Do not guarantee employment, hiring, interview success,
    or job acceptance.

13. Give one clear and practical next action.

14. The apply recommendation should clearly explain whether
    the user should consider applying now while learning
    the missing skills.

15. If supervisor feedback exists, improve the recommendation
    according to that feedback.

16. Keep all recommendations grounded in the provided
    SkillGap analysis.
"""