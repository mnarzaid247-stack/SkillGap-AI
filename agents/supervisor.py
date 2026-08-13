import os
from typing import Any, Literal

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError


load_dotenv()


# ==========================================
# 1. Structured Output Schema
# ==========================================

class SupervisorOutput(BaseModel):
    decision: Literal[
        "approve",
        "retry_profile",
        "retry_requirements",
        "retry_career_coach",
        "controlled_failure",
    ]
    feedback: str


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
# 3. Supervisor Agent
# ==========================================

class SupervisorAgent:
    """
    Review specialist-agent outputs and decide whether
    the responsible agent should be approved or retried.

    The Supervisor does NOT redo specialist work.
    """

    VALID_STAGES = {
        "profile",
        "requirements",
        "career_coach",
    }

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
            temperature=0,
        )

        self.structured_llm = (
            self.llm.with_structured_output(
                SupervisorOutput,
                include_raw=True,
            )
        )

    def review(
        self,
        state: dict,
    ) -> dict:
        """
        Review the current specialist output.

        Input:
            state["review_stage"]

        Outputs:
            state["supervisor_decision"]
            state["supervisor_feedback"]
            state["supervisor_error"]
        """

        review_stage = str(
            state.get(
                "review_stage",
                "",
            )
        ).strip()

        if review_stage not in self.VALID_STAGES:
            return {
                "supervisor_decision":
                    "controlled_failure",

                "supervisor_feedback": (
                    f"Unsupported review stage: "
                    f"{review_stage or 'missing'}"
                ),

                "supervisor_error": (
                    "Supervisor received an "
                    "unsupported review stage."
                ),
            }

        prompt_messages = self._build_prompt(
            state=state,
            review_stage=review_stage,
        )

        try:
            response = self.structured_llm.invoke(
                prompt_messages
            )

            result = response.get(
                "parsed"
            )

            parsing_error = response.get(
                "parsing_error"
            )

            if result is None:
                if parsing_error is not None:
                    raise parsing_error

                raise ValueError(
                    "Supervisor returned no parsed output."
                )

            validated = self._validate_decision(
                result=result,
                review_stage=review_stage,
            )

            return {
                "supervisor_decision":
                    validated.decision,

                "supervisor_feedback":
                    validated.feedback.strip(),

                "supervisor_error":
                    None,
            }

        except Exception as error:
            error_detail = (
                f"{type(error).__name__}"
            )

            if isinstance(
                error,
                ValidationError,
            ):
                try:
                    validation_errors = (
                        error.errors()
                    )

                    if validation_errors:
                        first_error = (
                            validation_errors[0]
                        )

                        location = ".".join(
                            str(part)
                            for part in first_error.get(
                                "loc",
                                [],
                            )
                        )

                        message = (
                            first_error.get(
                                "msg",
                                "Invalid structured output.",
                            )
                        )

                        if location:
                            error_detail = (
                                "ValidationError at "
                                f"'{location}': {message}"
                            )
                        else:
                            error_detail = (
                                "ValidationError: "
                                f"{message}"
                            )
                except Exception:
                    pass

            return {
                "supervisor_decision":
                    "controlled_failure",

                "supervisor_feedback": (
                    "Supervisor could not complete "
                    "the quality review."
                ),

                "supervisor_error": (
                    "Supervisor review failed: "
                    f"{error_detail}"
                ),
            }

    def _build_prompt(
        self,
        state: dict,
        review_stage: str,
    ) -> list:
        """
        Build stage-specific quality review instructions.
        """

        if review_stage == "profile":
            content_to_review = state.get(
                "candidate_profile",
                {},
            )

            source_context = {
                "cv_text": state.get(
                    "cv_text",
                    "",
                ),
            }

            allowed_retry = "retry_profile"

            criteria = """
Check whether the candidate profile is usable.

Review:
- Skills were extracted when clearly present.
- Projects were captured when present.
- Experience was captured when present.
- Education was captured when present.
- Skill evidence is grounded in the CV.
- The result is not empty or obviously incomplete.
- Unsupported candidate facts were not invented.
"""

        elif review_stage == "requirements":
            content_to_review = state.get(
                "job_requirements",
                {},
            )

            source_context = {
                "selected_job": state.get(
                    "selected_job",
                    {},
                ),
            }

            allowed_retry = (
                "retry_requirements"
            )

            criteria = """
Check whether the job requirements are usable
for deterministic gap analysis.

Review:
- Required skills were extracted when explicitly mandatory.
- Preferred skills were separated when possible.
- Frameworks and tools were captured when stated.
- Responsibilities were captured when available.
- Experience requirements were captured when available.
- Education requirements were captured when available.
- Requirements were not invented.
- There is enough structured information for gap analysis.
"""

        else:
            content_to_review = state.get(
                "recommendations",
                {},
            )

            source_context = {
                "gap_analysis": state.get(
                    "gap_analysis",
                    {},
                ),
            }

            allowed_retry = (
                "retry_career_coach"
            )

            criteria = """
Check whether the Career Coach output is useful
and grounded in the verified gap analysis.

Review:
- Priority gaps come only from verified
  missing required skills.
- No new missing skills were invented.
- Learning order is practical.
- The portfolio project addresses verified gaps.
- The next action is specific and realistic.
- Apply guidance does not guarantee employment.
- Skill Coverage is not described as hiring probability.
"""

        system_message = SystemMessage(
            content=(
                "You are the Supervisor Agent "
                "for SkillGap AI. "
                "\n\n"
                "Your job is quality review and routing. "
                "You do not perform the specialist agent's work. "
                "\n\n"
                "Review only whether the specialist output "
                "is sufficiently accurate, grounded, complete, "
                "and useful for the next workflow step. "
                "\n\n"
                "Do not add new CV facts, job requirements, "
                "skills, experience, recommendations, or "
                "career claims."
            )
        )

        human_message = HumanMessage(
            content=f"""
CURRENT REVIEW STAGE:
{review_stage}

SPECIALIST OUTPUT:
<specialist_output>
{content_to_review}
</specialist_output>

SOURCE CONTEXT:
<source_context>
{source_context}
</source_context>

QUALITY CRITERIA:
{criteria}

DECISION RULES:

1. If the specialist output is sufficiently usable,
   return:
   decision = "approve"

2. If the specialist output has a meaningful problem
   that the same specialist should correct, return:
   decision = "{allowed_retry}"

3. Use "controlled_failure" only when:
   - the output is unusable,
   - the source data is insufficient,
   - or safe review is not possible.

4. Do not request a retry for minor wording,
   formatting, or style preferences.

5. When returning a retry decision,
   feedback MUST clearly explain what needs improvement.

6. When returning approve,
   feedback may be concise or empty.

7. Never choose a retry decision belonging
   to another review stage.

Return only the structured response.
"""
        )

        return [
            system_message,
            human_message,
        ]

    def _validate_decision(
        self,
        result: SupervisorOutput,
        review_stage: str,
    ) -> SupervisorOutput:
        """
        Deterministically verify that the LLM returned
        a decision allowed for the current review stage.
        """

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

        if (
            result.decision
            not in allowed_decisions[
                review_stage
            ]
        ):
            return SupervisorOutput(
                decision="controlled_failure",
                feedback=(
                    "Supervisor returned an invalid "
                    f"decision '{result.decision}' "
                    f"for stage '{review_stage}'."
                ),
            )

        feedback = result.feedback.strip()

        if (
            result.decision.startswith(
                "retry_"
            )
            and not feedback
        ):
            return SupervisorOutput(
                decision="controlled_failure",
                feedback=(
                    "Supervisor requested a retry "
                    "without providing correction feedback."
                ),
            )

        return SupervisorOutput(
            decision=result.decision,
            feedback=feedback,
        )


# ==========================================
# 4. Local Test
# ==========================================

if __name__ == "__main__":
    test_state = {
        "review_stage": "career_coach",

        "gap_analysis": {
            "missing_required_skills": [
                "Docker",
                "SQL",
            ],
            "skill_coverage": 60.0,
        },

        "recommendations": {
            "priority_gaps": [
                {
                    "skill": "Docker",
                    "priority": "HIGH",
                    "reason": (
                        "Docker is a required "
                        "missing skill."
                    ),
                }
            ],
            "learning_order": [
                "Docker",
                "SQL",
            ],
            "portfolio_project": {
                "title": "AI API Project",
                "description": (
                    "Build and containerize "
                    "a small AI API."
                ),
                "technologies": [
                    "Python",
                    "Docker",
                    "SQL",
                ],
            },
            "next_action": (
                "Start learning Docker "
                "and apply it to a small project."
            ),
            "apply_recommendation": (
                "Consider applying while "
                "strengthening the missing skills."
            ),
        },
    }

    print(
        "--- Running Supervisor Agent ---"
    )

    supervisor = SupervisorAgent()

    output = supervisor.review(
        test_state
    )

    print("\nDecision:")
    print(
        output["supervisor_decision"]
    )

    print("\nFeedback:")
    print(
        output["supervisor_feedback"]
    )

    print("\nError:")
    print(
        output["supervisor_error"]
    )