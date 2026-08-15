from workflow.input_nodes import input_guard_node
from workflow.state import (
    MAX_CV_LENGTH,
    MAX_LOCATION_LENGTH,
    MAX_TARGET_ROLE_LENGTH,
)


def valid_state():
    return {
        "cv_text": (
            "Data analyst with experience in Python, SQL, "
            "Power BI, statistics, data analysis, and reporting."
        ),
        "target_role": "Data Analyst",
        "location": "Riyadh",
    }


def test_valid_input_passes():
    result = input_guard_node(
        valid_state()
    )

    assert "error_message" not in result
    assert result["target_role"] == "Data Analyst"
    assert result["location"] == "Riyadh"
    assert result["search_retries"] == 0
    assert result["profile_retry_count"] == 0


def test_empty_cv_is_rejected():
    state = valid_state()
    state["cv_text"] = ""

    result = input_guard_node(state)

    assert "error_message" in result
    assert "CV text is required." in result["error_message"]


def test_short_cv_is_rejected():
    state = valid_state()
    state["cv_text"] = "Too short"

    result = input_guard_node(state)

    assert "error_message" in result
    assert "CV text is too short." in result["error_message"]


def test_long_cv_is_rejected():
    state = valid_state()
    state["cv_text"] = "A" * (
        MAX_CV_LENGTH + 1
    )

    result = input_guard_node(state)

    assert "error_message" in result
    assert (
        "CV text exceeds the allowed length."
        in result["error_message"]
    )


def test_empty_target_role_is_rejected():
    state = valid_state()
    state["target_role"] = ""

    result = input_guard_node(state)

    assert "error_message" in result
    assert (
        "Target role is required."
        in result["error_message"]
    )


def test_long_target_role_is_rejected():
    state = valid_state()
    state["target_role"] = "A" * (
        MAX_TARGET_ROLE_LENGTH + 1
    )

    result = input_guard_node(state)

    assert "error_message" in result
    assert (
        "Target role exceeds the allowed length."
        in result["error_message"]
    )


def test_empty_location_is_rejected():
    state = valid_state()
    state["location"] = ""

    result = input_guard_node(state)

    assert "error_message" in result
    assert (
        "Location is required."
        in result["error_message"]
    )


def test_long_location_is_rejected():
    state = valid_state()
    state["location"] = "A" * (
        MAX_LOCATION_LENGTH + 1
    )

    result = input_guard_node(state)

    assert "error_message" in result
    assert (
        "Location exceeds the allowed length."
        in result["error_message"]
    )


def test_input_is_trimmed():
    state = valid_state()

    state["target_role"] = "  Data Analyst  "
    state["location"] = "  Riyadh  "

    result = input_guard_node(state)

    assert result["target_role"] == "Data Analyst"
    assert result["location"] == "Riyadh"