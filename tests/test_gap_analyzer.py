from agents.gap_analyzer import (
    GapMatchResult,
    SkillMatch,
    gap_analyzer,
)


class FakeStructuredLLM:
    def __init__(self, result):
        self.result = result

    def invoke(self, messages):
        return self.result


class FakeLLM:
    def __init__(self, result):
        self.result = result

    def with_structured_output(self, schema):
        return FakeStructuredLLM(
            self.result
        )


def base_state():
    return {
        "candidate_profile": {
            "skills": [
                "Python",
                "SQL",
                "Statistical Modeling",
                "Power BI",
            ],
            "skill_evidence": {
                "Python": (
                    "Used Python for data cleaning "
                    "and analysis projects."
                ),
                "SQL": (
                    "Used SQL for querying "
                    "structured datasets."
                ),
                "Statistical Modeling": (
                    "Built statistical forecasting "
                    "and regression models."
                ),
                "Power BI": (
                    "Created dashboards and reports."
                ),
            },
        },
        "job_requirements": {
            "required_skills": [
                "Python",
                "Statistical Analysis",
                "Machine Learning",
            ],
            "preferred_skills": [
                "Power BI",
            ],
        },
    }


def test_semantic_match_is_used():
    result = GapMatchResult(
        required_matches=[
            SkillMatch(
                requirement="Python",
                matched=True,
                evidence=[
                    "Used Python for data cleaning "
                    "and analysis projects."
                ],
                reason="Direct Python evidence.",
            ),
            SkillMatch(
                requirement="Statistical Analysis",
                matched=True,
                evidence=[
                    "Built statistical forecasting "
                    "and regression models."
                ],
                reason=(
                    "Statistical modeling and "
                    "forecasting demonstrate "
                    "statistical analysis capability."
                ),
            ),
            SkillMatch(
                requirement="Machine Learning",
                matched=False,
                evidence=[],
                reason=(
                    "No explicit machine learning "
                    "evidence was provided."
                ),
            ),
        ],
        preferred_matches=[
            SkillMatch(
                requirement="Power BI",
                matched=True,
                evidence=[
                    "Created dashboards and reports."
                ],
                reason="Direct Power BI evidence.",
            ),
        ],
    )

    output = gap_analyzer(
        base_state(),
        FakeLLM(result),
    )

    gap = output["gap_analysis"]

    assert output["gap_error"] is None

    assert (
        "Statistical Analysis"
        in gap["matching_required_skills"]
    )

    assert (
        "Machine Learning"
        in gap["missing_required_skills"]
    )


def test_python_does_not_automatically_match_machine_learning():
    result = GapMatchResult(
        required_matches=[
            SkillMatch(
                requirement="Python",
                matched=True,
                evidence=[
                    "Used Python for data analysis."
                ],
                reason="Direct Python evidence.",
            ),
            SkillMatch(
                requirement="Statistical Analysis",
                matched=True,
                evidence=[
                    "Built statistical models."
                ],
                reason="Supported by statistical work.",
            ),
            SkillMatch(
                requirement="Machine Learning",
                matched=False,
                evidence=[],
                reason=(
                    "Python alone does not prove "
                    "machine learning experience."
                ),
            ),
        ],
        preferred_matches=[],
    )

    output = gap_analyzer(
        base_state(),
        FakeLLM(result),
    )

    gap = output["gap_analysis"]

    assert (
        "Machine Learning"
        in gap["missing_required_skills"]
    )

    assert (
        "Machine Learning"
        not in gap["matching_required_skills"]
    )


def test_coverage_is_calculated_by_python():
    result = GapMatchResult(
        required_matches=[
            SkillMatch(
                requirement="Python",
                matched=True,
                evidence=["Python evidence"],
                reason="Matched.",
            ),
            SkillMatch(
                requirement="Statistical Analysis",
                matched=True,
                evidence=[
                    "Statistical modeling evidence"
                ],
                reason="Matched.",
            ),
            SkillMatch(
                requirement="Machine Learning",
                matched=False,
                evidence=[],
                reason="Not matched.",
            ),
        ],
        preferred_matches=[],
    )

    output = gap_analyzer(
        base_state(),
        FakeLLM(result),
    )

    gap = output["gap_analysis"]

    assert gap["skill_coverage"] == 66.7

    assert (
        gap["coverage_details"]["matched_required"]
        == 2
    )

    assert (
        gap["coverage_details"]["total_required"]
        == 3
    )


def test_missing_llm_requirement_is_treated_as_unmatched():
    result = GapMatchResult(
        required_matches=[
            SkillMatch(
                requirement="Python",
                matched=True,
                evidence=["Python evidence"],
                reason="Matched.",
            ),
        ],
        preferred_matches=[],
    )

    output = gap_analyzer(
        base_state(),
        FakeLLM(result),
    )

    gap = output["gap_analysis"]

    assert (
        "Statistical Analysis"
        in gap["missing_required_skills"]
    )

    assert (
        "Machine Learning"
        in gap["missing_required_skills"]
    )

    assert gap["skill_coverage"] == 33.3


def test_llm_cannot_add_new_requirement():
    result = GapMatchResult(
        required_matches=[
            SkillMatch(
                requirement="Python",
                matched=True,
                evidence=["Python evidence"],
                reason="Matched.",
            ),
            SkillMatch(
                requirement="Statistical Analysis",
                matched=True,
                evidence=[
                    "Statistical modeling evidence"
                ],
                reason="Matched.",
            ),
            SkillMatch(
                requirement="Machine Learning",
                matched=False,
                evidence=[],
                reason="Not matched.",
            ),
            SkillMatch(
                requirement="AWS",
                matched=True,
                evidence=["Fake AWS evidence"],
                reason="Should be ignored.",
            ),
        ],
        preferred_matches=[],
    )

    output = gap_analyzer(
        base_state(),
        FakeLLM(result),
    )

    gap = output["gap_analysis"]

    assert (
        "AWS"
        not in gap["matching_required_skills"]
    )

    assert (
        gap["coverage_details"]["total_required"]
        == 3
    )


def test_matched_skill_without_evidence_is_flagged():
    result = GapMatchResult(
        required_matches=[
            SkillMatch(
                requirement="Python",
                matched=True,
                evidence=[],
                reason="Matched but no evidence returned.",
            ),
            SkillMatch(
                requirement="Statistical Analysis",
                matched=True,
                evidence=[
                    "Statistical modeling evidence"
                ],
                reason="Matched.",
            ),
            SkillMatch(
                requirement="Machine Learning",
                matched=False,
                evidence=[],
                reason="Not matched.",
            ),
        ],
        preferred_matches=[],
    )

    output = gap_analyzer(
        base_state(),
        FakeLLM(result),
    )

    gap = output["gap_analysis"]

    assert (
        "Python"
        in gap["unsupported_required_skills"]
    )


def test_preferred_skill_does_not_affect_coverage():
    result = GapMatchResult(
        required_matches=[
            SkillMatch(
                requirement="Python",
                matched=True,
                evidence=["Python evidence"],
                reason="Matched.",
            ),
            SkillMatch(
                requirement="Statistical Analysis",
                matched=False,
                evidence=[],
                reason="Not matched.",
            ),
            SkillMatch(
                requirement="Machine Learning",
                matched=False,
                evidence=[],
                reason="Not matched.",
            ),
        ],
        preferred_matches=[
            SkillMatch(
                requirement="Power BI",
                matched=True,
                evidence=["Power BI evidence"],
                reason="Matched.",
            ),
        ],
    )

    output = gap_analyzer(
        base_state(),
        FakeLLM(result),
    )

    gap = output["gap_analysis"]

    assert gap["skill_coverage"] == 33.3

    assert (
        "Power BI"
        in gap["matching_preferred_skills"]
    )


def test_invalid_candidate_profile_returns_profile_error():
    state = base_state()

    state["candidate_profile"] = None

    result = GapMatchResult(
        required_matches=[],
        preferred_matches=[],
    )

    output = gap_analyzer(
        state,
        FakeLLM(result),
    )

    assert output["gap_error"] is not None
    assert (
        output["gap_error_source"]
        == "profile"
    )


def test_missing_required_skills_returns_requirements_error():
    state = base_state()

    state["job_requirements"][
        "required_skills"
    ] = []

    result = GapMatchResult(
        required_matches=[],
        preferred_matches=[],
    )

    output = gap_analyzer(
        state,
        FakeLLM(result),
    )

    assert output["gap_error"] is not None
    assert (
        output["gap_error_source"]
        == "requirements"
    )