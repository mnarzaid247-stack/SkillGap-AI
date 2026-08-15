from agents.gap_analyzer import gap_analyzer
from agents.requirements_agent import requirements_agent

from workflow.runtime import (
    _live_log,
    create_career_coach_agent,
    create_llm,
)
from workflow.state import MAX_PROFILE_RETRIES, MAX_REQUIREMENTS_RETRIES, SkillGapState


def requirements_node(
    state: SkillGapState,
) -> dict:

    _live_log("[START] Requirements Agent")

    result = requirements_agent(
    state,
    create_llm(),
)

    if result.get("requirements_error"):
        _live_log(
            f"[ERROR] Requirements Agent — {result.get('requirements_error')}"
        )
    else:
        _live_log("[DONE] Requirements Agent")

    return {
        **result,

        "execution_logs": [
            (
                "[Requirements Agent] "
                "Job requirements extracted"
            )
        ],
    }


def route_requirements_result(
    state: SkillGapState,
) -> str:

    if state.get("requirements_error"):
        retries = state.get(
            "requirements_retry_count",
            0,
        )

        if retries < MAX_REQUIREMENTS_RETRIES:
            return "retry"

        return "failure"

    return "supervisor"


def gap_analyzer_node(
    state: SkillGapState,
) -> dict:

    _live_log("[START] Gap Analyzer")

    result = gap_analyzer(
        state
    )

    coverage = (
        result
        .get(
            "gap_analysis",
            {},
        )
        .get(
            "skill_coverage"
        )
    )

    log_message = (
        "[Gap Analyzer] "
        "Skill gap analysis completed"
    )

    if coverage is not None:
        log_message += (
            f" — Skill Coverage = {coverage}%"
        )

    if result.get("gap_error"):
        _live_log(
            f"[ERROR] Gap Analyzer — {result.get('gap_error')}"
        )
    else:
        _live_log(
            f"[DONE] Gap Analyzer — Skill Coverage = {coverage}%"
            if coverage is not None
            else "[DONE] Gap Analyzer"
        )

    return {
        **result,

        "execution_logs": [
            log_message
        ],
    }


def route_gap_analysis(
    state: SkillGapState,
) -> str:
    """
    Deterministic quality routing.

    This is intentionally NOT an LLM decision.
    """

    error = state.get(
        "gap_error"
    )

    source = state.get(
        "gap_error_source"
    )

    if not error:
        return "career_coach"

    if source == "requirements":

        retries = state.get(
            "requirements_retry_count",
            0,
        )

        if retries < MAX_REQUIREMENTS_RETRIES:
            return "requirements_retry"

        return "controlled_failure"

    if source == "profile":

        retries = state.get(
            "profile_retry_count",
            0,
        )

        if retries < MAX_PROFILE_RETRIES:
            return "profile_repair"

        return "controlled_failure"

    return "controlled_failure"


def career_coach_node(
    state: SkillGapState,
) -> dict:

    _live_log("[START] Career Coach")

    result = (
    create_career_coach_agent()
    .generate_recommendations(
        state
    )
)

    if result.get("career_coach_error"):
        _live_log(
            f"[ERROR] Career Coach — {result.get('career_coach_error')}"
        )
    else:
        _live_log("[DONE] Career Coach")

    return {
        **result,

        "execution_logs": [
            (
                "[Career Coach] "
                "Career recommendations generated"
            )
        ],
    }
