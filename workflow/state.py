import operator
from typing import Annotated, TypedDict


MIN_VALID_JOBS = 3
MAX_DISPLAYED_JOBS = 5

MAX_SEARCH_RETRIES = 2
MAX_PROFILE_RETRIES = 1
MAX_REQUIREMENTS_RETRIES = 1
MAX_COACH_RETRIES = 1

MAX_CV_LENGTH = 20000
MAX_TARGET_ROLE_LENGTH = 100
MAX_LOCATION_LENGTH = 100


class SkillGapState(TypedDict, total=False):
    # User input
    cv_text: str
    target_role: str
    location: str

    # Profile Analyzer
    candidate_profile: dict
    profile_error: str | None
    profile_next_node: str

    # Job Scout
    jobs: list[dict]
    search_queries: list[str]
    search_retries: int
    job_scout_error: str | None
    valid_job_count: int
    limited_results: bool
    job_validation_error: str | None

    # Human-in-the-Loop
    selected_job_index: int
    selected_job: dict
    selected_job_text: str
    human_decision: str
    human_selection_error: str | None

    # Requirements Agent
    job_requirements: dict
    requirements_error: str | None

    # Gap Analyzer
    gap_analysis: dict
    gap_error: str | None
    gap_error_source: str | None

    # Career Coach
    recommendations: dict
    career_coach_error: str | None

    # Supervisor
    review_stage: str
    supervisor_decision: str
    supervisor_feedback: str
    supervisor_error: str | None

    # Retry counters
    profile_retry_count: int
    requirements_retry_count: int
    coach_retry_count: int

    # Parallel branch readiness
    profile_ready: bool
    jobs_ready: bool

    # Final output
    final_report: str
    error_message: str

    # Observability
    execution_logs: Annotated[list[str], operator.add]