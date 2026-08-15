from agents.profile_analyzer import profile_analyzer_agent

from workflow.runtime import _live_log
from workflow.state import (
    MAX_CV_LENGTH,
    MAX_LOCATION_LENGTH,
    MAX_TARGET_ROLE_LENGTH,
    SkillGapState,
)


def input_guard_node(
    state: SkillGapState,
) -> dict:

    _live_log("[START] Input Guard")

    cv_text = str(
        state.get("cv_text", "")
    ).strip()

    target_role = str(
        state.get("target_role", "")
    ).strip()

    location = str(
        state.get("location", "")
    ).strip()

    errors = []

    if not cv_text:
        errors.append(
            "CV text is required."
        )

    elif len(cv_text) < 50:
        errors.append(
            "CV text is too short."
        )

    elif len(cv_text) > MAX_CV_LENGTH:
        errors.append(
            "CV text exceeds the allowed length."
        )

    if not target_role:
        errors.append(
            "Target role is required."
        )

    elif len(target_role) > MAX_TARGET_ROLE_LENGTH:
        errors.append(
            "Target role exceeds the allowed length."
        )

    if not location:
        errors.append(
            "Location is required."
        )

    elif len(location) > MAX_LOCATION_LENGTH:
        errors.append(
            "Location exceeds the allowed length."
        )

    if errors:
        _live_log(
            f"[ERROR] Input Guard — {' '.join(errors)}"
        )

        return {
            "error_message": " ".join(errors),
            "execution_logs": [
                "[Input Guard] Input validation failed"
            ],
        }

    _live_log("[DONE] Input Guard")

    return {
        "cv_text": cv_text,
        "target_role": target_role,
        "location": location,

        "search_retries": 0,
        "profile_retry_count": 0,
        "requirements_retry_count": 0,
        "coach_retry_count": 0,

        "limited_results": False,

        "execution_logs": [
            "[Input Guard] Input validated"
        ],
    }


def route_input_guard(
    state: SkillGapState,
) -> str:

    if state.get("error_message"):
        return "controlled_failure"

    return "parallel_start"


def parallel_start_node(
    state: SkillGapState,
) -> dict:
    """
    Fan-out point.

    Profile Analyzer and Job Scout start
    independently after this node.
    """

    _live_log(
        "[DONE] Workflow Fan-Out — Profile Analyzer + Job Scout"
    )

    return {
        "execution_logs": [
            (
                "[Workflow] Starting Profile Analyzer "
                "and Job Scout in parallel"
            )
        ]
    }


def profile_analyzer_node(
    state: SkillGapState,
) -> dict:

    _live_log("[START] Profile Analyzer")

    result = profile_analyzer_agent(
        state
    )

    if result.get("profile_error"):
        _live_log(
            f"[ERROR] Profile Analyzer — "
            f"{result.get('profile_error')}"
        )
    else:
        _live_log("[DONE] Profile Analyzer")

    return {
        **result,

        "execution_logs": [
            "[Profile Analyzer] CV analyzed"
        ],
    }


def profile_ready_node(
    state: SkillGapState,
) -> dict:

    _live_log("[DONE] Profile Branch Ready")

    return {
        "profile_ready": True,

        "execution_logs": [
            "[Workflow] Candidate profile approved"
        ],
    }
