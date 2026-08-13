import operator
import os
import re
import uuid
from typing import Annotated, TypedDict
from urllib.parse import urlparse

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from tavily import TavilyClient

from agents.profile_analyzer import profile_analyzer_agent
from agents.job_scout import job_scout_agent
from agents.requirements_agent import requirements_agent
from agents.gap_analyzer import gap_analyzer
from agents.career_coach import CareerCoachAgent
from agents.supervisor import SupervisorAgent


load_dotenv()


# ==========================================
# 1. Constants
# ==========================================

MIN_VALID_JOBS = 3
MAX_DISPLAYED_JOBS = 5

MAX_SEARCH_RETRIES = 2
MAX_PROFILE_RETRIES = 1
MAX_REQUIREMENTS_RETRIES = 1
MAX_COACH_RETRIES = 1

MAX_CV_LENGTH = 20000


# ==========================================
# 2. Shared State
# ==========================================

class SkillGapState(TypedDict, total=False):

    # --------------------------------------
    # User Input
    # --------------------------------------

    cv_text: str
    target_role: str
    location: str

    # --------------------------------------
    # Profile Analyzer
    # --------------------------------------

    candidate_profile: dict
    profile_error: str | None

    # Used when Profile Agent is retried
    # from a later stage.
    profile_next_node: str

    # --------------------------------------
    # Job Scout
    # --------------------------------------

    jobs: list[dict]
    search_queries: list[str]
    search_retries: int
    job_scout_error: str | None

    valid_job_count: int
    limited_results: bool
    job_validation_error: str | None

    # --------------------------------------
    # Human-in-the-Loop
    # --------------------------------------

    selected_job_index: int
    selected_job: dict
    selected_job_text: str

    human_decision: str
    human_selection_error: str | None

    # --------------------------------------
    # Requirements Agent
    # --------------------------------------

    job_requirements: dict
    requirements_error: str | None

    # --------------------------------------
    # Gap Analyzer
    # --------------------------------------

    gap_analysis: dict
    gap_error: str | None
    gap_error_source: str | None

    # --------------------------------------
    # Career Coach
    # --------------------------------------

    recommendations: dict
    career_coach_error: str | None

    # --------------------------------------
    # Supervisor
    # --------------------------------------

    review_stage: str
    supervisor_decision: str
    supervisor_feedback: str
    supervisor_error: str | None

    # --------------------------------------
    # Retry Counters
    # --------------------------------------

    profile_retry_count: int
    requirements_retry_count: int
    coach_retry_count: int

    # --------------------------------------
    # Parallel Branch Readiness
    # --------------------------------------

    profile_ready: bool
    jobs_ready: bool

    # --------------------------------------
    # Final Output
    # --------------------------------------

    final_report: str
    error_message: str

    # --------------------------------------
    # Observability
    # --------------------------------------

    execution_logs: Annotated[
        list[str],
        operator.add,
    ]


# ==========================================
# 3. LLM
# ==========================================

def create_llm() -> ChatOpenAI:
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

    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.1,
    )


llm = create_llm()

career_coach_agent = CareerCoachAgent()
supervisor_agent = SupervisorAgent()


# ==========================================
# Live Terminal Logging
# ==========================================

def _live_log(message: str) -> None:
    """Print workflow progress immediately in the terminal."""
    print(message, flush=True)


# ==========================================
# 4. Input Guard
# ==========================================

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

    if not location:
        errors.append(
            "Location is required."
        )

    if errors:
        _live_log(f"[ERROR] Input Guard — {' '.join(errors)}")
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


# ==========================================
# 5. Parallel Start
# ==========================================

def parallel_start_node(
    state: SkillGapState,
) -> dict:
    """
    Fan-out point.

    Profile Analyzer and Job Scout start
    independently after this node.
    """

    _live_log("[DONE] Workflow Fan-Out — Profile Analyzer + Job Scout")

    return {
        "execution_logs": [
            (
                "[Workflow] Starting Profile Analyzer "
                "and Job Scout in parallel"
            )
        ]
    }


# ==========================================
# 6. Profile Analyzer Node
# ==========================================

def profile_analyzer_node(
    state: SkillGapState,
) -> dict:

    _live_log("[START] Profile Analyzer")

    result = profile_analyzer_agent(
        state
    )

    if result.get("profile_error"):
        _live_log(
            f"[ERROR] Profile Analyzer — {result.get('profile_error')}"
        )
    else:
        _live_log("[DONE] Profile Analyzer")

    return {
        **result,

        "execution_logs": [
            "[Profile Analyzer] CV analyzed"
        ],
    }


# ==========================================
# 7. Job Scout Node
# ==========================================

def job_scout_node(
    state: SkillGapState,
) -> dict:

    _live_log("[START] Job Scout")

    result = job_scout_agent(
        state
    )

    if result.get("job_scout_error"):
        _live_log(
            f"[ERROR] Job Scout — {result.get('job_scout_error')}"
        )
    else:
        _live_log(
            f"[DONE] Job Scout — {len(result.get('jobs', []))} raw results"
        )

    return {
        **result,

        "execution_logs": [
            (
                "[Job Scout] Searching current "
                f"{state.get('target_role', '')} jobs "
                f"in {state.get('location', '')}"
            )
        ],
    }


# ==========================================
# 8. Job Validation
# ==========================================

def _normalize_text(
    text: str,
) -> str:

    value = str(
        text
    ).casefold()

    value = re.sub(
        r"[^\w+#. ]+",
        " ",
        value,
        flags=re.UNICODE,
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    return value.strip()


def _valid_job_url(
    url: str,
) -> bool:

    try:
        parsed = urlparse(
            str(url)
        )

        return (
            parsed.scheme
            in {"http", "https"}
            and bool(parsed.netloc)
        )

    except Exception:
        return False


def _job_looks_relevant(
    job: dict,
    target_role: str,
) -> bool:
    """
    Conservative deterministic relevance check.

    This is intentionally simple because the
    project is a one-day capstone.
    """

    title = _normalize_text(
        job.get("title", "")
    )

    description = _normalize_text(
        job.get("description", "")
    )

    if not title:
        return False

    role = _normalize_text(
        target_role
    )

    role_words = {
        word
        for word in role.split()
        if len(word) > 1
    }

    combined = (
        f"{title} {description}"
    )

    # Exact or partial target-role match
    if role and role in combined:
        return True

    if role_words:
        overlap = sum(
            1
            for word in role_words
            if word in combined
        )

        if overlap >= max(
            1,
            len(role_words) // 2,
        ):
            return True

    # Controlled related AI titles
    ai_terms = {
        "ai",
        "artificial intelligence",
        "machine learning",
        "ml",
        "generative ai",
        "genai",
        "llm",
    }

    target_is_ai = any(
        term in role
        for term in ai_terms
    )

    if target_is_ai:
        return any(
            term in combined
            for term in ai_terms
        )

    return False


def _job_is_explicitly_closed(
    job: dict,
) -> bool:

    text = _normalize_text(
        (
            f"{job.get('title', '')} "
            f"{job.get('description', '')}"
        )
    )

    closed_terms = [
        "job expired",
        "position expired",
        "position closed",
        "applications closed",
        "no longer accepting applications",
        "vacancy closed",
    ]

    return any(
        term in text
        for term in closed_terms
    )


def job_validation_node(
    state: SkillGapState,
) -> dict:

    _live_log("[START] Job Validation")

    jobs = state.get(
        "jobs",
        [],
    )

    target_role = state.get(
        "target_role",
        "",
    )

    if not isinstance(
        jobs,
        list,
    ):
        jobs = []

    valid_jobs = []
    seen_urls = set()

    for job in jobs:

        if not isinstance(
            job,
            dict,
        ):
            continue

        url = str(
            job.get(
                "url",
                "",
            )
        ).strip()

        title = str(
            job.get(
                "title",
                "",
            )
        ).strip()

        if not title:
            continue

        if not _valid_job_url(
            url
        ):
            continue

        if url in seen_urls:
            continue

        if _job_is_explicitly_closed(
            job
        ):
            continue

        if not _job_looks_relevant(
            job,
            target_role,
        ):
            continue

        seen_urls.add(
            url
        )

        valid_jobs.append(
            job
        )

    valid_jobs = valid_jobs[
        :MAX_DISPLAYED_JOBS
    ]

    _live_log(
        f"[DONE] Job Validation — {len(valid_jobs)} valid jobs"
    )

    return {
        "jobs": valid_jobs,

        "valid_job_count":
            len(valid_jobs),

        "job_validation_error":
            None,

        "execution_logs": [
            (
                "[Job Validation] "
                f"{len(valid_jobs)} valid jobs found"
            )
        ],
    }


def route_job_validation(
    state: SkillGapState,
) -> str:

    count = state.get(
        "valid_job_count",
        0,
    )

    retries = state.get(
        "search_retries",
        0,
    )

    # Ideal result
    if count >= MIN_VALID_JOBS:
        return "jobs_ready"

    # Retry search while allowed
    if retries < MAX_SEARCH_RETRIES:
        return "refine_search"

    # Limited fallback
    if 1 <= count < MIN_VALID_JOBS:
        return "limited_jobs_ready"

    # Zero usable results
    return "controlled_failure"


# ==========================================
# 9. Search Retry Loop
# ==========================================

def refine_search_node(
    state: SkillGapState,
) -> dict:

    current = state.get(
        "search_retries",
        0,
    )

    new_count = (
        current + 1
    )

    _live_log(
        f"[RETRY] Job Search — {new_count}/{MAX_SEARCH_RETRIES}"
    )

    return {
        "search_retries":
            new_count,

        "execution_logs": [
            (
                "[Search Refinement] "
                f"Retry {new_count}/"
                f"{MAX_SEARCH_RETRIES}"
            )
        ],
    }


# ==========================================
# 10. Parallel Branch Completion
# ==========================================

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


def jobs_ready_node(
    state: SkillGapState,
) -> dict:

    _live_log("[DONE] Job Search Ready")

    return {
        "jobs_ready": True,
        "limited_results": False,

        "execution_logs": [
            "[Workflow] Job search ready"
        ],
    }


def limited_jobs_ready_node(
    state: SkillGapState,
) -> dict:

    _live_log("[DONE] Job Search Ready — limited results")

    return {
        "jobs_ready": True,
        "limited_results": True,

        "execution_logs": [
            (
                "[Workflow] Continuing with "
                "limited job results"
            )
        ],
    }


def jobs_branch_done_node(
    state: SkillGapState,
) -> dict:
    """
    Common completion point for the job-search branch.

    Both normal job results and limited job results pass
    through this node before the Human-in-the-Loop fan-in.

    This preserves the existing value of limited_results.
    """

    _live_log("[DONE] Job Search Branch")

    return {
        "execution_logs": [
            "[Workflow] Job-search branch completed"
        ],
    }


# ==========================================
# 11. Human-in-the-Loop
# ==========================================

def human_job_selection_node(
    state: SkillGapState,
) -> dict:

    _live_log("[WAIT] Human Job Selection")

    jobs = state.get(
        "jobs",
        [],
    )

    display_jobs = []

    for index, job in enumerate(
        jobs,
        start=1,
    ):
        display_jobs.append(
            {
                "number": index,
                "title": job.get(
                    "title",
                    "Not listed",
                ),
                "company": job.get(
                    "company",
                    "Not listed",
                ),
                "location": job.get(
                    "location",
                    "Not listed",
                ),
                "source": job.get(
                    "source",
                    "",
                ),
                "url": job.get(
                    "url",
                    "",
                ),
            }
        )

    selection = interrupt(
        {
            "message": (
                "Select one job to analyze."
            ),
            "limited_results":
                state.get(
                    "limited_results",
                    False,
                ),
            "jobs": display_jobs,
        }
    )

    if isinstance(
        selection,
        dict,
    ):
        selection = selection.get(
            "selected_job_index"
        )

    try:
        selected_number = int(
            selection
        )

    except (
        TypeError,
        ValueError,
    ):
        return {
            "human_selection_error": (
                "Job selection must be a number."
            ),

            "execution_logs": [
                "[Human Review] Invalid job selection"
            ],
        }

    if not (
        1
        <= selected_number
        <= len(jobs)
    ):
        return {
            "human_selection_error": (
                "Selected job number is out of range."
            ),

            "execution_logs": [
                "[Human Review] Invalid job selection"
            ],
        }

    selected_index = (
        selected_number - 1
    )

    selected_job = jobs[
        selected_index
    ]

    return {
        "selected_job_index":
            selected_index,

        "selected_job":
            selected_job,

        "human_decision":
            "selected",

        "human_selection_error":
            None,

        "execution_logs": [
            (
                "[Human Review] "
                f"Job #{selected_number} selected"
            )
        ],
    }


def route_human_selection(
    state: SkillGapState,
) -> str:

    if state.get(
        "human_selection_error"
    ):
        return "human_job_selection"

    if state.get(
        "selected_job"
    ):
        return "selected_job_enrichment"

    return "controlled_failure"


# ==========================================
# 12. Selected Job Enrichment
# ==========================================

def selected_job_enrichment_node(
    state: SkillGapState,
) -> dict:
    """
    Try to extract richer content from the selected
    job URL before the Requirements Agent runs.

    If extraction fails, keep the original Tavily
    search snippet instead of fabricating data.
    """

    _live_log("[START] Selected Job Enrichment")

    selected_job = dict(
        state.get(
            "selected_job",
            {},
        )
    )

    url = str(
        selected_job.get(
            "url",
            "",
        )
    ).strip()

    existing_description = str(
        selected_job.get(
            "description",
            "",
        )
    ).strip()

    if not url:
        return {
            "selected_job":
                selected_job,

            "selected_job_text":
                existing_description,

            "execution_logs": [
                (
                    "[Job Extract] No URL available; "
                    "using search result content"
                )
            ],
        }

    api_key = os.getenv(
        "TAVILY_API_KEY"
    )

    if not api_key:
        return {
            "selected_job":
                selected_job,

            "selected_job_text":
                existing_description,

            "execution_logs": [
                (
                    "[Job Extract] Tavily key missing; "
                    "using search result content"
                )
            ],
        }

    try:
        client = TavilyClient(
            api_key=api_key
        )

        response = client.extract(
            url,
            extract_depth="basic",
        )

        results = response.get(
            "results",
            [],
        )

        extracted_text = ""

        if results:
            extracted_text = str(
                results[0].get(
                    "raw_content",
                    "",
                )
            ).strip()

        if extracted_text:
            selected_job[
                "description"
            ] = extracted_text

            return {
                "selected_job":
                    selected_job,

                "selected_job_text":
                    extracted_text,

                "execution_logs": [
                    (
                        "[Job Extract] Selected job "
                        "content extracted"
                    )
                ],
            }

    except Exception as error:
        return {
            "selected_job":
                selected_job,

            "selected_job_text":
                existing_description,

            "execution_logs": [
                (
                    "[Job Extract] Extraction failed "
                    f"({type(error).__name__}); "
                    "using search result content"
                )
            ],
        }

    return {
        "selected_job":
            selected_job,

        "selected_job_text":
            existing_description,

        "execution_logs": [
            (
                "[Job Extract] No additional content; "
                "using search result content"
            )
        ],
    }


# ==========================================
# 13. Requirements Agent
# ==========================================

def requirements_node(
    state: SkillGapState,
) -> dict:

    _live_log("[START] Requirements Agent")

    result = requirements_agent(
        state,
        llm,
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


# ==========================================
# 14. Gap Analyzer
# ==========================================

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


# ==========================================
# 15. Career Coach
# ==========================================

def career_coach_node(
    state: SkillGapState,
) -> dict:

    _live_log("[START] Career Coach")

    result = (
        career_coach_agent
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


# ==========================================
# 16. Supervisor
# ==========================================

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


# ==========================================
# 17. Supervisor Routing
# ==========================================

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


# ==========================================
# 18. Retry Nodes
# ==========================================

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


# ==========================================
# 19. Gap Quality Routing
# ==========================================

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


# ==========================================
# 20. Final Report Helpers
# ==========================================

def _format_list(
    items: list,
    prefix: str = "-",
) -> str:

    if not items:
        return "None"

    return "\n".join(
        f"{prefix} {item}"
        for item in items
    )


def _format_priority_gaps(
    gaps: list[dict],
) -> str:

    if not gaps:
        return "No critical required-skill gaps identified."

    lines = []

    for index, gap in enumerate(
        gaps,
        start=1,
    ):
        lines.append(
            (
                f"{index}. "
                f"{gap.get('skill', 'Unknown')} "
                f"— {gap.get('priority', 'Unknown')}\n"
                f"   {gap.get('reason', '')}"
            )
        )

    return "\n".join(
        lines
    )


def _format_evidence_gaps(
    gaps: list[dict],
) -> str:

    if not gaps:
        return "No evidence gaps detected."

    lines = []

    for gap in gaps:
        lines.append(
            (
                f"- {gap.get('skill', 'Unknown')}: "
                f"{gap.get('reason', '')}"
            )
        )

    return "\n".join(
        lines
    )


# ==========================================
# 21. Final Report
# ==========================================

def final_report_node(
    state: SkillGapState,
) -> dict:

    _live_log("[START] Final Report")

    profile = state.get(
        "candidate_profile",
        {},
    )

    requirements = state.get(
        "job_requirements",
        {},
    )

    gap = state.get(
        "gap_analysis",
        {},
    )

    recommendations = state.get(
        "recommendations",
        {},
    )

    selected_job = state.get(
        "selected_job",
        {},
    )

    portfolio = recommendations.get(
        "portfolio_project",
        {},
    )

    report = f"""
SKILLGAP AI
Career Opportunity Analysis

Target Role:
{state.get("target_role", "Not provided")}

Location:
{state.get("location", "Not provided")}

Selected Opportunity:
{selected_job.get("title", "Not listed")}
Company: {selected_job.get("company") or "Not listed"}

==================================================

PROFILE SUMMARY

{profile.get("summary", "Not available")}

Detected Skills:
{_format_list(profile.get("skills", []), "✓")}

Experience Level:
{profile.get("experience_level", "Unknown")}

==================================================

JOB REQUIREMENTS

Required Skills:
{_format_list(requirements.get("required_skills", []))}

Preferred Skills:
{_format_list(requirements.get("preferred_skills", []))}

Frameworks / Tools:
{_format_list(requirements.get("frameworks", []))}

==================================================

STRONG MATCHES

{_format_list(gap.get("matching_skills", []), "✓")}

==================================================

MISSING REQUIRED SKILLS

{_format_list(gap.get("missing_required_skills", []), "✗")}

==================================================

EVIDENCE GAPS

{_format_evidence_gaps(gap.get("evidence_gaps", []))}

==================================================

SKILL COVERAGE

{gap.get("skill_coverage", 0)}%

Matched Required Skills:
{gap.get("coverage_details", {}).get("matched_required", 0)}

Total Required Skills:
{gap.get("coverage_details", {}).get("total_required", 0)}

==================================================

TOP PRIORITY GAPS

{_format_priority_gaps(recommendations.get("priority_gaps", []))}

==================================================

RECOMMENDED LEARNING ORDER

{_format_list(recommendations.get("learning_order", []))}

==================================================

RECOMMENDED PORTFOLIO PROJECT

Title:
{portfolio.get("title", "Not available")}

Description:
{portfolio.get("description", "Not available")}

Technologies:
{_format_list(portfolio.get("technologies", []))}

==================================================

NEXT ACTION

{recommendations.get("next_action", "Not available")}

==================================================

APPLY RECOMMENDATION

{recommendations.get("apply_recommendation", "Not available")}
"""

    _live_log("[DONE] Final Report")

    return {
        "final_report":
            report.strip(),

        "execution_logs": [
            "[Final Report] Report generated"
        ],
    }


# ==========================================
# 22. Controlled Failure
# ==========================================

def controlled_failure_node(
    state: SkillGapState,
) -> dict:

    message = (
        state.get("error_message")
        or state.get("profile_error")
        or state.get("job_scout_error")
        or state.get("job_validation_error")
        or state.get("human_selection_error")
        or state.get("requirements_error")
        or state.get("gap_error")
        or state.get("career_coach_error")
        or state.get("supervisor_error")
        or state.get("supervisor_feedback")
        or (
            "Workflow stopped because the "
            "analysis could not be validated."
        )
    )

    _live_log(f"[ERROR] Controlled Failure — {message}")

    return {
        "error_message":
            message,

        "execution_logs": [
            (
                "[Controlled Failure] "
                f"{message}"
            )
        ],
    }


# ==========================================
# 23. Graph Builder
# ==========================================
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

def build_graph():

    workflow = StateGraph(
        SkillGapState
    )

    # --------------------------------------
    # Nodes
    # --------------------------------------

    workflow.add_node(
        "input_guard",
        input_guard_node,
    )

    workflow.add_node(
        "parallel_start",
        parallel_start_node,
    )

    workflow.add_node(
        "profile_analyzer",
        profile_analyzer_node,
    )

    workflow.add_node(
        "job_scout",
        job_scout_node,
    )

    workflow.add_node(
        "job_validation",
        job_validation_node,
    )

    workflow.add_node(
        "refine_search",
        refine_search_node,
    )

    workflow.add_node(
        "profile_ready",
        profile_ready_node,
    )

    workflow.add_node(
        "jobs_ready",
        jobs_ready_node,
    )

    workflow.add_node(
        "limited_jobs_ready",
        limited_jobs_ready_node,
    )

    workflow.add_node(
        "jobs_branch_done",
        jobs_branch_done_node,
    )

    workflow.add_node(
        "human_job_selection",
        human_job_selection_node,
    )

    workflow.add_node(
        "selected_job_enrichment",
        selected_job_enrichment_node,
    )

    workflow.add_node(
        "requirements",
        requirements_node,
    )

    workflow.add_node(
        "gap_analyzer",
        gap_analyzer_node,
    )

    workflow.add_node(
        "career_coach",
        career_coach_node,
    )

    workflow.add_node(
        "supervisor",
        supervisor_node,
    )

    workflow.add_node(
        "profile_retry",
        profile_retry_node,
    )

    workflow.add_node(
        "requirements_retry",
        requirements_retry_node,
    )

    workflow.add_node(
        "career_coach_retry",
        career_coach_retry_node,
    )

    workflow.add_node(
        "profile_repair",
        profile_repair_node,
    )

    workflow.add_node(
        "final_report",
        final_report_node,
    )

    workflow.add_node(
        "controlled_failure",
        controlled_failure_node,
    )

    # --------------------------------------
    # Start
    # --------------------------------------

    workflow.add_edge(
        START,
        "input_guard",
    )

    workflow.add_conditional_edges(
        "input_guard",
        route_input_guard,
        {
            "parallel_start":
                "parallel_start",

            "controlled_failure":
                "controlled_failure",
        },
    )

    # ======================================
    # PARALLEL FAN-OUT
    # ======================================

    workflow.add_edge(
        "parallel_start",
        "profile_analyzer",
    )

    workflow.add_edge(
        "parallel_start",
        "job_scout",
    )

    # --------------------------------------
    # Profile Branch
    # --------------------------------------

    workflow.add_edge(
        "profile_analyzer",
        "supervisor",
    )

    # --------------------------------------
    # Job Search Branch
    # --------------------------------------

    workflow.add_edge(
        "job_scout",
        "job_validation",
    )

    workflow.add_conditional_edges(
        "job_validation",
        route_job_validation,
        {
            "jobs_ready":
                "jobs_ready",

            "limited_jobs_ready":
                "limited_jobs_ready",

            "refine_search":
                "refine_search",

            "controlled_failure":
                "controlled_failure",
        },
    )

    workflow.add_edge(
        "refine_search",
        "job_scout",
    )

    workflow.add_edge(
        "jobs_ready",
        "jobs_branch_done",
    )

    workflow.add_edge(
        "limited_jobs_ready",
        "jobs_branch_done",
    )

    # --------------------------------------
    # Supervisor Routing
    # --------------------------------------

    workflow.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "profile_ready":
                "profile_ready",

            "gap_analyzer":
                "gap_analyzer",

            "final_report":
                "final_report",

            "profile_retry":
                "profile_retry",

            "requirements_retry":
                "requirements_retry",

            "career_coach_retry":
                "career_coach_retry",

            "controlled_failure":
                "controlled_failure",
        },
    )

    workflow.add_edge(
        "profile_retry",
        "profile_analyzer",
    )

    # ======================================
    # PARALLEL FAN-IN
    # ======================================
    #
    # The workflow waits for BOTH:
    # 1. approved candidate profile
    # 2. validated job results
    #
    # before Human-in-the-Loop starts.
    # ======================================

    workflow.add_edge(
        [
            "profile_ready",
            "jobs_branch_done",
        ],
        "human_job_selection",
    )

    # --------------------------------------
    # Human Selection
    # --------------------------------------

    workflow.add_conditional_edges(
        "human_job_selection",
        route_human_selection,
        {
            "human_job_selection":
                "human_job_selection",

            "selected_job_enrichment":
                "selected_job_enrichment",

            "controlled_failure":
                "controlled_failure",
        },
    )

    workflow.add_edge(
    "selected_job_enrichment",
    "requirements",
)

# --------------------------------------
# Requirements
# --------------------------------------

    workflow.add_conditional_edges(
    "requirements",
    route_requirements_result,
    {
        "supervisor": "supervisor",
        "retry": "requirements_retry",
        "failure": "controlled_failure",
    },
)

    workflow.add_edge(
    "requirements_retry",
    "requirements",
)

    # --------------------------------------
    # Gap Analyzer / Quality Check
    # --------------------------------------

    workflow.add_conditional_edges(
        "gap_analyzer",
        route_gap_analysis,
        {
            "career_coach":
                "career_coach",

            "requirements_retry":
                "requirements_retry",

            "profile_repair":
                "profile_repair",

            "controlled_failure":
                "controlled_failure",
        },
    )

    workflow.add_edge(
        "profile_repair",
        "profile_analyzer",
    )

    # --------------------------------------
    # Career Coach
    # --------------------------------------

    workflow.add_edge(
        "career_coach",
        "supervisor",
    )

    workflow.add_edge(
        "career_coach_retry",
        "career_coach",
    )

    # --------------------------------------
    # End
    # --------------------------------------

    workflow.add_edge(
        "final_report",
        END,
    )

    workflow.add_edge(
        "controlled_failure",
        END,
    )

    # --------------------------------------
    # Checkpointer required for interrupt()
    # --------------------------------------

    checkpointer = InMemorySaver()

    return workflow.compile(
        checkpointer=checkpointer
    )


graph = build_graph()


# ==========================================
# 24. CLI Demo
# ==========================================

def _print_jobs_from_interrupt(
    interrupt_value,
):
    """
    Print the job list sent by the
    Human-in-the-Loop node.
    """

    if not isinstance(
        interrupt_value,
        dict,
    ):
        return

    jobs = interrupt_value.get(
        "jobs",
        [],
    )

    print(
        "\n=== CURRENT OPPORTUNITIES ==="
    )

    if interrupt_value.get(
        "limited_results"
    ):
        print(
            "\nWarning: Only a limited number "
            "of valid jobs were found.\n"
        )

    for job in jobs:
        print(
            f"\n{job['number']}. "
            f"{job['title']}"
        )

        print(
            "   Company:",
            job.get(
                "company"
            ) or "Not listed",
        )

        print(
            "   Location:",
            job.get(
                "location"
            ) or "Not listed",
        )

        print(
            "   Source:",
            job.get(
                "source"
            ) or "Not listed",
        )


def run_cli():

    print(
        "\n================================"
    )
    print(
        "           SKILLGAP AI"
    )
    print(
        "================================\n"
    )

    cv_path = input(
        "CV text file path: "
    ).strip()

    try:
        with open(
            cv_path,
            "r",
            encoding="utf-8",
        ) as file:
            cv_text = file.read()

    except Exception as error:
        print(
            "\nCould not read CV file:",
            type(error).__name__,
        )
        return

    target_role = input(
        "Target role: "
    ).strip()

    location = input(
        "Location: "
    ).strip()

    initial_state = {
        "cv_text":
            cv_text,

        "target_role":
            target_role,

        "location":
            location,

        "execution_logs":
            [],
    }

    thread_id = str(
        uuid.uuid4()
    )

    config = {
        "configurable": {
            "thread_id":
                thread_id,
        }
    }

    result = graph.invoke(
        initial_state,
        config=config,
    )

    # --------------------------------------
    # Human-in-the-Loop resume cycle
    # --------------------------------------

    while "__interrupt__" in result:

        interrupts = result[
            "__interrupt__"
        ]

        if not interrupts:
            break

        current_interrupt = (
            interrupts[0]
        )

        interrupt_value = getattr(
            current_interrupt,
            "value",
            current_interrupt,
        )

        _print_jobs_from_interrupt(
            interrupt_value
        )

        selection = input(
            "\nSelect job number: "
        ).strip()

        result = graph.invoke(
            Command(
                resume=selection
            ),
            config=config,
        )

    # --------------------------------------
    # Output
    # --------------------------------------

    if result.get(
        "error_message"
    ):
        print(
            "\n=== WORKFLOW ERROR ===\n"
        )

        print(
            result[
                "error_message"
            ]
        )

    elif result.get(
        "final_report"
    ):
        print(
            "\n\n"
            + result[
                "final_report"
            ]
        )

    print(
        "\n\n=== EXECUTION LOGS ==="
    )

    for log in result.get(
        "execution_logs",
        [],
    ):
        print(
            log
        )


# ==========================================
# 25. Entry Point
# ==========================================

if __name__ == "__main__":
    run_cli()