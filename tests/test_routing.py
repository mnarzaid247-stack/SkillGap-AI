from workflow.analysis_nodes import (
    route_gap_analysis,
    route_requirements_result,
)
from workflow.input_nodes import (
    route_input_guard,
)
from workflow.job_selection_nodes import (
    route_human_selection,
)

from workflow.job_search_nodes import (
    route_job_validation,
)
from workflow.state import (
    MAX_PROFILE_RETRIES,
    MAX_REQUIREMENTS_RETRIES,
    MAX_SEARCH_RETRIES,
)
from workflow.supervision_nodes import (
    route_supervisor,
)


def test_input_guard_routes_valid_input():
    state = {
        "error_message": "",
    }

    assert (
        route_input_guard(state)
        == "parallel_start"
    )


def test_input_guard_routes_error_to_failure():
    state = {
        "error_message": "Invalid input",
    }

    assert (
        route_input_guard(state)
        == "controlled_failure"
    )


def test_job_validation_routes_enough_jobs():
    state = {
        "valid_job_count": 3,
        "search_retries": 0,
    }

    assert (
        route_job_validation(state)
        == "jobs_ready"
    )


def test_job_validation_routes_to_retry():
    state = {
        "valid_job_count": 1,
        "search_retries": 0,
    }

    assert (
        route_job_validation(state)
        == "refine_search"
    )


def test_job_validation_routes_limited_results():
    state = {
        "valid_job_count": 2,
        "search_retries":
            MAX_SEARCH_RETRIES,
    }

    assert (
        route_job_validation(state)
        == "limited_jobs_ready"
    )


def test_job_validation_routes_zero_results_to_failure():
    state = {
        "valid_job_count": 0,
        "search_retries":
            MAX_SEARCH_RETRIES,
    }

    assert (
        route_job_validation(state)
        == "controlled_failure"
    )


def test_human_selection_routes_valid_selection():
    state = {
        "selected_job": {
            "title": "Data Analyst"
        },
        "human_selection_error": None,
    }

    assert (
        route_human_selection(state)
        == "selected_job_enrichment"
    )


def test_human_selection_routes_invalid_selection_back():
    state = {
        "human_selection_error":
            "Invalid selection",
    }

    assert (
        route_human_selection(state)
        == "human_job_selection"
    )


def test_human_selection_routes_missing_selection_to_failure():
    state = {
        "human_selection_error": None,
    }

    assert (
        route_human_selection(state)
        == "controlled_failure"
    )


def test_requirements_success_routes_to_supervisor():
    state = {
        "requirements_error": None,
    }

    assert (
        route_requirements_result(state)
        == "supervisor"
    )


def test_requirements_error_routes_to_retry():
    state = {
        "requirements_error":
            "Invalid requirements",
        "requirements_retry_count": 0,
    }

    assert (
        route_requirements_result(state)
        == "retry"
    )


def test_requirements_error_routes_to_failure_after_limit():
    state = {
        "requirements_error":
            "Invalid requirements",
        "requirements_retry_count":
            MAX_REQUIREMENTS_RETRIES,
    }

    assert (
        route_requirements_result(state)
        == "failure"
    )


def test_gap_success_routes_to_career_coach():
    state = {
        "gap_error": None,
    }

    assert (
        route_gap_analysis(state)
        == "career_coach"
    )


def test_gap_requirements_error_routes_to_retry():
    state = {
        "gap_error":
            "Requirements problem",
        "gap_error_source":
            "requirements",
        "requirements_retry_count": 0,
    }

    assert (
        route_gap_analysis(state)
        == "requirements_retry"
    )


def test_gap_requirements_error_fails_after_retry_limit():
    state = {
        "gap_error":
            "Requirements problem",
        "gap_error_source":
            "requirements",
        "requirements_retry_count":
            MAX_REQUIREMENTS_RETRIES,
    }

    assert (
        route_gap_analysis(state)
        == "controlled_failure"
    )


def test_gap_profile_error_routes_to_profile_repair():
    state = {
        "gap_error":
            "Profile problem",
        "gap_error_source":
            "profile",
        "profile_retry_count": 0,
    }

    assert (
        route_gap_analysis(state)
        == "profile_repair"
    )


def test_gap_profile_error_fails_after_retry_limit():
    state = {
        "gap_error":
            "Profile problem",
        "gap_error_source":
            "profile",
        "profile_retry_count":
            MAX_PROFILE_RETRIES,
    }

    assert (
        route_gap_analysis(state)
        == "controlled_failure"
    )


def test_supervisor_approved_profile_routes_to_profile_ready():
    state = {
        "supervisor_decision":
            "approve",
        "review_stage":
            "profile",
    }

    assert (
        route_supervisor(state)
        == "profile_ready"
    )


def test_supervisor_approved_profile_uses_next_node():
    state = {
        "supervisor_decision":
            "approve",
        "review_stage":
            "profile",
        "profile_next_node":
            "gap_analyzer",
    }

    assert (
        route_supervisor(state)
        == "gap_analyzer"
    )


def test_supervisor_approved_requirements_routes_to_gap():
    state = {
        "supervisor_decision":
            "approve",
        "review_stage":
            "requirements",
    }

    assert (
        route_supervisor(state)
        == "gap_analyzer"
    )


def test_supervisor_approved_career_coach_routes_to_report():
    state = {
        "supervisor_decision":
            "approve",
        "review_stage":
            "career_coach",
    }

    assert (
        route_supervisor(state)
        == "final_report"
    )


def test_supervisor_unknown_decision_routes_to_failure():
    state = {
        "supervisor_decision":
            "unknown",
        "review_stage":
            "profile",
    }

    assert (
        route_supervisor(state)
        == "controlled_failure"
    )