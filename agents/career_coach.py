import os
from typing import Any, Literal

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


load_dotenv()


# ==========================================
# 1. Structured Output Schemas
# ==========================================

class PriorityGap(BaseModel):
    skill: str
    priority: Literal["HIGH", "MEDIUM", "LOW"]
    reason: str


class PortfolioProject(BaseModel):
    title: str
    description: str
    technologies: list[str] = Field(
        default_factory=list
    )


class CareerCoachOutput(BaseModel):
    priority_gaps: list[PriorityGap] = Field(
        default_factory=list,
        description=(
            "Up to three highest-priority verified "
            "missing required skills."
        ),
    )

    learning_order: list[str] = Field(
        default_factory=list,
        description=(
            "Recommended order for learning verified "
            "missing required skills."
        ),
    )

    portfolio_project: PortfolioProject

    next_action: str

    apply_recommendation: str


# ==========================================
# 2. Helper
# ==========================================

def _model_to_dict(
    model: BaseModel,
) -> dict[str, Any]:
    """Support Pydantic versions 1 and 2."""

    if hasattr(model, "model_dump"):
        return model.model_dump()

    return model.dict()


# ==========================================
# 3. Career Coach Agent
# ==========================================

class CareerCoachAgent:
    """
    Convert verified gap-analysis results into
    practical career recommendations.

    The Career Coach does NOT:
    - re-analyze the CV
    - re-analyze the job
    - calculate Skill Coverage
    - create new missing skills
    """

    def __init__(self):
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

        self.llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0.2,
        )

        self.structured_llm = (
            self.llm.with_structured_output(
                CareerCoachOutput
            )
        )

    def generate_recommendations(
        self,
        state: dict,
    ) -> dict:
        """
        Generate career recommendations.

        Inputs:
            state["target_role"]
            state["gap_analysis"]
            state["supervisor_feedback"]

        Outputs:
            state["recommendations"]
            state["career_coach_error"]
            state["review_stage"]
        """

        gap_analysis = state.get(
            "gap_analysis"
        )

        if not isinstance(
            gap_analysis,
            dict,
        ) or not gap_analysis:
            return {
                "recommendations": {},
                "career_coach_error": (
                    "Gap analysis is missing or invalid."
                ),
                "review_stage": "career_coach",
            }

        missing_skills = gap_analysis.get(
            "missing_required_skills",
            [],
        )

        matched_skills = gap_analysis.get(
            "matching_required_skills",
            [],
        )

        evidence_gaps = gap_analysis.get(
            "evidence_gaps",
            [],
        )

        skill_coverage = gap_analysis.get(
            "skill_coverage"
        )

        if not isinstance(
            missing_skills,
            list,
        ):
            return {
                "recommendations": {},
                "career_coach_error": (
                    "Missing required skills "
                    "have an invalid format."
                ),
                "review_stage": "career_coach",
            }

        if not isinstance(
            matched_skills,
            list,
        ):
            matched_skills = []

        if not isinstance(
            evidence_gaps,
            list,
        ):
            evidence_gaps = []

        if not isinstance(
            skill_coverage,
            (int, float),
        ):
            return {
                "recommendations": {},
                "career_coach_error": (
                    "Skill Coverage is missing "
                    "or invalid."
                ),
                "review_stage": "career_coach",
            }

        prompt_messages = (
            self._build_prompt(
                state=state,
                gap_analysis=gap_analysis,
            )
        )

        try:
            result: CareerCoachOutput = (
                self.structured_llm.invoke(
                    prompt_messages
                )
            )

            recommendations = (
                _model_to_dict(result)
            )

            # Deterministic grounding check:
            # priority gaps must be verified missing skills.
            allowed_missing = {
                str(skill).casefold().strip()
                for skill in missing_skills
            }

            for gap in recommendations.get(
                "priority_gaps",
                [],
            ):
                skill = str(
                    gap.get("skill", "")
                ).casefold().strip()

                if skill not in allowed_missing:
                    return {
                        "recommendations": {},
                        "career_coach_error": (
                            "Career Coach generated "
                            "an unverified priority gap."
                        ),
                        "review_stage":
                            "career_coach",
                    }

            return {
                "recommendations":
                    recommendations,

                "career_coach_error":
                    None,

                "review_stage":
                    "career_coach",
            }

        except Exception as error:
            return {
                "recommendations": {},
                "career_coach_error": (
                    f"Career Coach failed: "
                    f"{type(error).__name__}"
                ),
                "review_stage":
                    "career_coach",
            }

    def _build_prompt(
        self,
        state: dict,
        gap_analysis: dict,
    ) -> list:
        """Build grounded Career Coach messages."""

        target_role = str(
            state.get(
                "target_role",
                "",
            )
        ).strip()

        matched_skills = (
            gap_analysis.get(
                "matching_required_skills",
                [],
            )
        )

        missing_skills = (
            gap_analysis.get(
                "missing_required_skills",
                [],
            )
        )

        missing_preferred = (
            gap_analysis.get(
                "missing_preferred_skills",
                [],
            )
        )

        evidence_gaps = (
            gap_analysis.get(
                "evidence_gaps",
                [],
            )
        )

        skill_coverage = (
            gap_analysis.get(
                "skill_coverage",
                0,
            )
        )

        existing_priority_gaps = (
            gap_analysis.get(
                "priority_gaps",
                [],
            )
        )

        supervisor_feedback = str(
            state.get(
                "supervisor_feedback",
                "",
            )
        ).strip()

        feedback_section = ""

        if supervisor_feedback:
            feedback_section = f"""
Supervisor feedback from the previous recommendation:
<supervisor_feedback>
{supervisor_feedback}
</supervisor_feedback>

Use this feedback only to improve the recommendations.
Do not create new gaps or facts to satisfy the feedback.
"""

        system_message = SystemMessage(
            content=(
                "You are the Career Coach Agent "
                "for SkillGap AI. "
                "\n\n"
                "Your responsibility is to turn "
                "VERIFIED gap-analysis results into "
                "practical career recommendations. "
                "\n\n"
                "You are NOT responsible for "
                "re-analyzing the CV, job advertisement, "
                "or Skill Coverage score. "
                "\n\n"
                "Use only the verified information "
                "provided to you. "
                "Never invent missing skills, experience, "
                "job requirements, or career facts. "
                "\n\n"
                "Do not guarantee employment, interviews, "
                "job acceptance, or hiring success."
            )
        )

        human_message = HumanMessage(
            content=f"""
TARGET ROLE:
{target_role or "Not provided"}

MATCHED REQUIRED SKILLS:
{matched_skills}

MISSING REQUIRED SKILLS:
{missing_skills}

MISSING PREFERRED SKILLS:
{missing_preferred}

EVIDENCE GAPS:
{evidence_gaps}

SKILL COVERAGE:
{skill_coverage}%

VERIFIED PRIORITY GAP DATA:
{existing_priority_gaps}

{feedback_section}

Recommendation rules:

1. Return at most THREE priority gaps.

2. Every priority gap MUST come from
   MISSING REQUIRED SKILLS.

3. Do not introduce a skill that is not present
   in MISSING REQUIRED SKILLS.

4. Prioritize missing required skills before
   preferred skills or evidence gaps.

5. Use evidence gaps as supporting career advice,
   but do not falsely label an evidence gap as a
   missing skill.

6. Recommend a realistic learning order using
   verified missing required skills.

7. If there are no missing required skills,
   priority_gaps and learning_order may be empty.

8. Suggest exactly ONE realistic portfolio project.

9. When missing required skills exist, the project
   should help demonstrate one or more of them.

10. The project may also use already matched skills.

11. Do not claim the user knows a technology unless
    it appears in the verified analysis.

12. Give one clear practical next action.

13. The apply recommendation must explain whether
    applying while learning the gaps is reasonable.

14. Never describe Skill Coverage as:
    - hiring probability
    - acceptance probability
    - chance of getting the job.

15. Keep the recommendations concise, useful,
    and grounded in the verified analysis.
"""
        )

        return [
            system_message,
            human_message,
        ]


# ==========================================
# 4. Local Test
# ==========================================

if __name__ == "__main__":
    test_state = {
        "target_role": "AI Engineer",

        "gap_analysis": {
            "matching_required_skills": [
                "Python",
                "RAG",
                "Git",
            ],

            "missing_required_skills": [
                "Docker",
                "SQL",
            ],

            "missing_preferred_skills": [
                "AWS",
            ],

            "evidence_gaps": [
                {
                    "skill":
                        "Prompt Engineering",

                    "reason":
                        "Skill is listed in the CV "
                        "but lacks clear project evidence.",
                }
            ],

            "skill_coverage": 60.0,

            "priority_gaps": [
                {
                    "skill": "Docker",
                    "priority": "HIGH",
                    "reason":
                        "Required skill is missing "
                        "from the CV.",
                },
                {
                    "skill": "SQL",
                    "priority": "HIGH",
                    "reason":
                        "Required skill is missing "
                        "from the CV.",
                },
            ],
        },

        "supervisor_feedback": "",
    }

    print(
        "--- Running Career Coach Agent ---"
    )

    agent = CareerCoachAgent()

    output = agent.generate_recommendations(
        test_state
    )

    print("\nRecommendations:")
    print(output["recommendations"])

    print("\nCareer Coach Error:")
    print(output["career_coach_error"])

    print("\nReview Stage:")
    print(output["review_stage"])