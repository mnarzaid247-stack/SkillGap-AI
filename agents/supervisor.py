import os
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI


load_dotenv()


class SupervisorOutput(BaseModel):
    decision: Literal[
        "approve",
        "retry_profile",
        "retry_requirements",
        "retry_career_coach",
        "controlled_failure",
    ]
    feedback: str


class SupervisorAgent:
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
            temperature=0.0,
        )

        self.structured_llm = self.llm.with_structured_output(
            SupervisorOutput
        )

    def review(self, state: dict) -> SupervisorOutput:
        review_stage = state.get("review_stage", "")

        if review_stage not in {
            "profile",
            "requirements",
            "career_coach",
        }:
            return SupervisorOutput(
                decision="controlled_failure",
                feedback=f"Unsupported review stage: {review_stage}",
            )

        prompt = self._build_prompt(state, review_stage)

        result = self.structured_llm.invoke(prompt)

        return self._validate_decision(
            result=result,
            review_stage=review_stage,
        )

    def _build_prompt(
        self,
        state: dict,
        review_stage: str,
    ) -> str:

        if review_stage == "profile":
            content_to_review = state.get("profile", {})

            allowed_retry = "retry_profile"

            criteria = """
Review whether the profile analysis is usable.

Check that:
- Technical skills were extracted when present.
- Projects were captured when present.
- Experience and education were captured when present.
- Skill evidence is grounded in the CV.
- The output is not empty or obviously incomplete.
- The output does not invent unsupported information.
"""

        elif review_stage == "requirements":
            content_to_review = state.get(
                "job_requirements",
                {}
            )

            allowed_retry = "retry_requirements"

            criteria = """
Review whether the selected job requirements were extracted
well enough for gap analysis.

Check that:
- Required skills are present when stated in the job.
- Preferred skills are separated when possible.
- Responsibilities are captured when available.
- Experience requirements are captured when available.
- Education requirements are captured when available.
- The output does not invent requirements.
- The result contains enough information for meaningful comparison.
"""

        else:
            content_to_review = state.get(
                "recommendations",
                {}
            )

            allowed_retry = "retry_career_coach"

            gap_analysis = state.get(
                "gap_analysis",
                {}
            )

            criteria = f"""
Review whether the Career Coach recommendations are useful
and grounded in the verified gap analysis.

VERIFIED GAP ANALYSIS:
{gap_analysis}

Check that:
- Priority gaps come from verified missing required skills.
- No new missing skills were invented.
- Recommendations are practical.
- The learning order is reasonable.
- The portfolio project addresses verified gaps.
- The next action is specific.
- The apply recommendation does not guarantee employment.
"""

        return f"""
You are the Supervisor Agent for SkillGap AI.

Your responsibility is quality review and routing.

CURRENT REVIEW STAGE:
{review_stage}

CONTENT TO REVIEW:
{content_to_review}

QUALITY CRITERIA:
{criteria}

RULES:

1. Do NOT redo the work of the specialist agent.

2. Do NOT add new profile facts, job requirements,
   missing skills, or career claims.

3. Review only the quality, completeness, consistency,
   and grounding of the provided output.

4. If the output is acceptable, return:
   decision = "approve"

5. If the output needs improvement, return only:
   decision = "{allowed_retry}"

6. Use "controlled_failure" only when the output is unusable
   or cannot be safely reviewed.

7. The feedback must clearly explain what should be improved.

8. Do not request retries for minor wording preferences.

9. Focus on problems that materially affect downstream analysis.

Return only the structured response.
"""

    def _validate_decision(
        self,
        result: SupervisorOutput,
        review_stage: str,
    ) -> SupervisorOutput:

        allowed_decisions = {
            "profile": {
                "approve",
                "retry_profile",
                "controlled_failure",
            },
            "requirements": {
                "approve",
                "retry_requirements",
                "controlled_failure",
            },
            "career_coach": {
                "approve",
                "retry_career_coach",
                "controlled_failure",
            },
        }

        if result.decision not in allowed_decisions[review_stage]:
            return SupervisorOutput(
                decision="controlled_failure",
                feedback=(
                    f"Invalid supervisor decision "
                    f"'{result.decision}' for stage "
                    f"'{review_stage}'."
                ),
            )

        return result