import os
import re
from urllib.parse import urlparse

from langgraph.types import interrupt
from tavily import TavilyClient

from agents.job_scout import job_scout_agent
from workflow.runtime import _live_log
from workflow.state import (
    MAX_DISPLAYED_JOBS,
    MAX_SEARCH_RETRIES,
    MIN_VALID_JOBS,
    SkillGapState,
)


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
