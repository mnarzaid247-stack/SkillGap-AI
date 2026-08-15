import re
from urllib.parse import urlparse

from agents.job_scout import job_scout_agent
from workflow.runtime import _live_log
from workflow.state import (
    MAX_DISPLAYED_JOBS,
    MAX_SEARCH_RETRIES,
    MIN_VALID_JOBS,
    SkillGapState,
)


def _merge_jobs(
    existing_jobs: list,
    new_jobs: list,
) -> list[dict]:
    """
    Preserve previously validated jobs across search retries
    and append newly discovered results.

    Deduplication is URL-based.
    """

    merged_jobs = []
    seen_urls = set()

    for collection in (
        existing_jobs,
        new_jobs,
    ):
        if not isinstance(
            collection,
            list,
        ):
            continue

        for job in collection:
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

            if not url:
                continue

            if url in seen_urls:
                continue

            seen_urls.add(
                url
            )

            merged_jobs.append(
                job
            )

    return merged_jobs


def job_scout_node(
    state: SkillGapState,
) -> dict:

    _live_log("[START] Job Scout")

    previous_jobs = state.get(
        "jobs",
        [],
    )

    result = job_scout_agent(
        state
    )

    new_jobs = result.get(
        "jobs",
        [],
    )

    merged_jobs = _merge_jobs(
        existing_jobs=previous_jobs,
        new_jobs=new_jobs,
    )

    if result.get(
        "job_scout_error"
    ):
        _live_log(
            f"[ERROR] Job Scout — "
            f"{result.get('job_scout_error')}"
        )
    else:
        _live_log(
            "[DONE] Job Scout — "
            f"{len(new_jobs)} raw results, "
            f"{len(merged_jobs)} total with preserved jobs"
        )

    return {
        **result,

        "jobs":
            merged_jobs,

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
        r"[^\w+#. ?؟/-]+",
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
        "انتهى التقديم",
        "التقديم مغلق",
        "لم يعد التقديم متاح",
    ]

    return any(
        term in text
        for term in closed_terms
    )


def _job_is_listing_page(
    job: dict,
) -> bool:
    """
    Detect pages that aggregate multiple job postings.
    """

    title = _normalize_text(
        job.get("title", "")
    )

    url = str(
        job.get("url", "")
    ).casefold()

    listing_patterns = [
        r"\b\d+\s+وظيفة",
        r"\b\d+\s+وظائف",
        r"\b\d+\s+jobs?\b",
        r"\b\d+\s+vacancies\b",
        r"وظائف\s+.+\s+في\s+السعودية",
        r"وظائف\s+.+\s+في\s+الرياض",
        r"وظائف\s+.+\s+السعودية",
        r"jobs?\s+.+\s+in\s+saudi arabia",
        r"jobs?\s+.+\s+in\s+riyadh",
        r"job search",
        r"search results",
        r"open positions",
        r"current openings",
    ]

    if any(
        re.search(
            pattern,
            title,
        )
        for pattern in listing_patterns
    ):
        return True

    listing_url_patterns = [
        "/job-search/",
        "/job-search",
        "/jobs/search",
        "/search/jobs",
        "/search-jobs",
        "/jobs-in-",
        "/vacancies/",
        "/vacancies?",
    ]

    if any(
        pattern in url
        for pattern in listing_url_patterns
    ):
        return True

    return False


def _job_is_article_or_guidance_page(
    job: dict,
) -> bool:
    """
    Reject articles, educational pages, career guides,
    learning paths, salary pages, and question-style content.
    """

    title = _normalize_text(
        job.get("title", "")
    )

    url = str(
        job.get("url", "")
    ).casefold()

    if not title:
        return True

    if "?" in title or "؟" in title:
        return True

    article_title_patterns = [
        r"^هل\s",
        r"^ما\s+هو\s",
        r"^ما\s+هي\s",
        r"^كيف\s",
        r"^لماذا\s",
        r"^what\s+is\s",
        r"^what\s+are\s",
        r"^how\s+to\s",
        r"^why\s",
        r"\bcareer guide\b",
        r"\bcareer path\b",
        r"\blearning path\b",
        r"\broadmap\b",
        r"\bsalary guide\b",
        r"\bsalary\b",
        r"\bcourse\b",
        r"\bcourses\b",
        r"\btutorial\b",
        r"\bbootcamp\b",
        r"\bdegree\b",
        r"\bmajor\b",
        r"\bcertification\b",
        r"\bدليل\b",
        r"\bمسار تعليمي\b",
        r"\bمسار مهني\b",
        r"\bرواتب\b",
        r"\bراتب\b",
        r"\bدورة\b",
        r"\bدورات\b",
        r"\bتخصص\b",
        r"\bتعلم\b",
        r"\bشهادة\b",
    ]

    if any(
        re.search(
            pattern,
            title,
        )
        for pattern in article_title_patterns
    ):
        return True

    article_url_patterns = [
        "/blog/",
        "/blogs/",
        "/article/",
        "/articles/",
        "/career-advice/",
        "/career-guide/",
        "/guides/",
        "/guide/",
        "/learn/",
        "/learning/",
        "/courses/",
        "/course/",
        "/salary/",
        "/salaries/",
    ]

    if any(
        pattern in url
        for pattern in article_url_patterns
    ):
        return True

    return False


def _job_has_specific_posting_signals(
    job: dict,
) -> bool:
    """
    Require basic signals that the result can plausibly
    represent one job posting.

    Company is intentionally NOT mandatory because Tavily
    does not always expose it reliably.
    """

    title = _normalize_text(
        job.get("title", "")
    )

    description = _normalize_text(
        job.get("description", "")
    )

    url = str(
        job.get("url", "")
    ).strip()

    if not title or not url:
        return False

    if len(title) < 3:
        return False

    if _job_is_listing_page(
        job
    ):
        return False

    if _job_is_article_or_guidance_page(
        job
    ):
        return False

    # Search snippets for real job postings usually contain
    # at least some usable text. Keep this threshold low so
    # legitimate sparse results are not discarded.
    if description and len(description) < 20:
        return False

    return True


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

        if not _job_has_specific_posting_signals(
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
        f"[DONE] Job Validation — "
        f"{len(valid_jobs)} valid jobs"
    )

    return {
        "jobs":
            valid_jobs,

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

    if count >= MIN_VALID_JOBS:
        return "jobs_ready"

    if retries < MAX_SEARCH_RETRIES:
        return "refine_search"

    if 1 <= count < MIN_VALID_JOBS:
        return "limited_jobs_ready"

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
        f"[RETRY] Job Search — "
        f"{new_count}/{MAX_SEARCH_RETRIES}"
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

    _live_log(
        "[DONE] Job Search Ready"
    )

    return {
        "jobs_ready":
            True,

        "limited_results":
            False,

        "execution_logs": [
            "[Workflow] Job search ready"
        ],
    }


def limited_jobs_ready_node(
    state: SkillGapState,
) -> dict:

    _live_log(
        "[DONE] Job Search Ready — "
        "limited results"
    )

    return {
        "jobs_ready":
            True,

        "limited_results":
            True,

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
    """

    _live_log(
        "[DONE] Job Search Branch"
    )

    return {
        "execution_logs": [
            "[Workflow] Job-search branch completed"
        ],
    }