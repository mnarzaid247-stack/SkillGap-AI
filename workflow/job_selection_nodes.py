import os

from langgraph.types import interrupt
from tavily import TavilyClient

from workflow.runtime import _live_log
from workflow.state import SkillGapState


def human_job_selection_node(
    state: SkillGapState,
) -> dict:

    _live_log(
        "[WAIT] Human Job Selection"
    )

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
                "number":
                    index,

                "title":
                    job.get(
                        "title",
                        "Not listed",
                    ),

                "company":
                    job.get(
                        "company",
                        "Not listed",
                    ),

                "location":
                    job.get(
                        "location",
                        "Not listed",
                    ),

                "source":
                    job.get(
                        "source",
                        "",
                    ),

                "url":
                    job.get(
                        "url",
                        "",
                    ),
            }
        )

    selection = interrupt(
        {
            "message":
                "Select one job to analyze.",

            "limited_results":
                state.get(
                    "limited_results",
                    False,
                ),

            "jobs":
                display_jobs,
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

    _live_log(
        "[START] Selected Job Enrichment"
    )

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