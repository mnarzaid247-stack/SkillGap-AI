from workflow.runtime import _live_log, supervisor_agent
from workflow.state import (
    MAX_COACH_RETRIES,
    MAX_PROFILE_RETRIES,
    MAX_REQUIREMENTS_RETRIES,
    SkillGapState,
)


def supervisor_node(
    state: SkillGapState,
) -> dict:

    stage = state.get(
        "review_stage",
        "unknown",
    )

    _live_log(f"[START] Supervisor [{stage}]")

    result = (
        supervisor_agent.review(
            state
        )
    )

    decision = result.get(
        "supervisor_decision",
        "controlled_failure",
    )

    _live_log(
        f"[DONE] Supervisor [{stage}] — {decision}"
    )

    return {
        **result,

        "execution_logs": [
            (
                f"[Supervisor] {stage} "
                f"review → {decision}"
            )
        ],
    }


def route_supervisor(
    state: SkillGapState,
) -> str:

    decision = state.get(
        "supervisor_decision"
    )

    review_stage = state.get(
        "review_stage"
    )

    # --------------------------------------
    # Approved
    # --------------------------------------

    if decision == "approve":

        if review_stage == "profile":
            return state.get(
                "profile_next_node",
                "profile_ready",
            )

        if review_stage == "requirements":
            return "gap_analyzer"

        if review_stage == "career_coach":
            return "final_report"

        return "controlled_failure"

    # --------------------------------------
    # Retry Profile
    # --------------------------------------

    if decision == "retry_profile":

        retries = state.get(
            "profile_retry_count",
            0,
        )

        if retries < MAX_PROFILE_RETRIES:
            return "profile_retry"

        return "controlled_failure"

    # --------------------------------------
    # Retry Requirements
    # --------------------------------------

    if decision == "retry_requirements":

        retries = state.get(
            "requirements_retry_count",
            0,
        )

        if retries < MAX_REQUIREMENTS_RETRIES:
            return "requirements_retry"

        return "controlled_failure"

    # --------------------------------------
    # Retry Career Coach
    # --------------------------------------

    if decision == "retry_career_coach":

        retries = state.get(
            "coach_retry_count",
            0,
        )

        if retries < MAX_COACH_RETRIES:
            return "career_coach_retry"

        return "controlled_failure"

    return "controlled_failure"


def profile_retry_node(
    state: SkillGapState,
) -> dict:

    count = (
        state.get(
            "profile_retry_count",
            0,
        )
        + 1
    )

    _live_log("[RETRY] Profile Analyzer — gap quality repair")

    return {
        "profile_retry_count":
            count,

        "execution_logs": [
            (
                "[Retry] Profile Analyzer "
                f"{count}/{MAX_PROFILE_RETRIES}"
            )
        ],
    }


def requirements_retry_node(
    state: SkillGapState,
) -> dict:

    count = (
        state.get(
            "requirements_retry_count",
            0,
        )
        + 1
    )

    _live_log(
        f"[RETRY] Requirements Agent — {count}/{MAX_REQUIREMENTS_RETRIES}"
    )

    return {
        "requirements_retry_count":
            count,

        "execution_logs": [
            (
                "[Retry] Requirements Agent "
                f"{count}/{MAX_REQUIREMENTS_RETRIES}"
            )
        ],
    }


def career_coach_retry_node(
    state: SkillGapState,
) -> dict:

    count = (
        state.get(
            "coach_retry_count",
            0,
        )
        + 1
    )

    _live_log(
        f"[RETRY] Career Coach — {count}/{MAX_COACH_RETRIES}"
    )

    return {
        "coach_retry_count":
            count,

        "execution_logs": [
            (
                "[Retry] Career Coach "
                f"{count}/{MAX_COACH_RETRIES}"
            )
        ],
    }


def profile_repair_node(
    state: SkillGapState,
) -> dict:
    """
    If the Gap Analyzer discovers a profile problem,
    retry only Profile Analyzer.

    After Supervisor approval, return directly
    to Gap Analyzer instead of restarting search/HITL.
    """

    count = (
        state.get(
            "profile_retry_count",
            0,
        )
        + 1
    )

    return {
        "profile_retry_count":
            count,

        "profile_next_node":
            "gap_analyzer",

        "execution_logs": [
            (
                "[Quality Check] Profile issue detected; "
                "retrying Profile Analyzer only"
            )
        ],
    }
