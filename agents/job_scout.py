import os
from datetime import datetime, timezone
from urllib.parse import urlparse

from dotenv import load_dotenv
from tavily import TavilyClient


load_dotenv()


# ==========================================
# 1. Constants
# ==========================================

MAX_RESULTS_PER_SEARCH = 5


# ==========================================
# 2. Helper Functions
# ==========================================

def _is_valid_url(url: str) -> bool:
    """Check whether the result contains a valid HTTP/HTTPS URL."""

    if not url:
        return False

    parsed = urlparse(url)

    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _extract_company_from_title(title: str) -> str | None:
    """
    Try to extract company name from common search-result titles.

    Examples:
        "AI Engineer - Company Name"
        "AI Engineer | Company Name"
        "AI Engineer at Company Name"

    If the company cannot be identified safely, return None.
    """

    if not title:
        return None

    separators = [
        " at ",
        " - ",
        " | ",
    ]

    for separator in separators:
        if separator in title:
            parts = title.split(separator)

            if len(parts) >= 2:
                company = parts[-1].strip()

                if company:
                    return company

    return None


def _deduplicate_jobs(jobs: list[dict]) -> list[dict]:
    """Remove duplicate jobs using their URLs."""

    unique_jobs = []
    seen_urls = set()

    for job in jobs:
        url = job.get("url", "").strip()

        if not url or url in seen_urls:
            continue

        seen_urls.add(url)
        unique_jobs.append(job)

    return unique_jobs


def _build_search_queries(
    target_role: str,
    location: str,
    retry_count: int,
) -> list[str]:
    """
    Build controlled search queries.

    The query becomes broader when the search workflow retries.
    """

    if retry_count <= 0:
        return [
            f'"{target_role}" jobs {location}',
            f'"{target_role}" careers {location}',
        ]

    if retry_count == 1:
        return [
            f'{target_role} jobs {location}',
            f'{target_role} vacancy {location}',
            f'AI Engineer jobs {location}',
            f'Generative AI Engineer jobs {location}',
            f'AI Specialist jobs {location}',
        ]

    return [
        f'{target_role} jobs Saudi Arabia',
        f'AI Engineer jobs Saudi Arabia',
        f'Generative AI Engineer jobs Saudi Arabia',
        f'AI Specialist jobs Saudi Arabia',
        f'Junior AI Engineer jobs Saudi Arabia',
    ]


# ==========================================
# 3. Job Scout Agent
# ==========================================

def job_scout_agent(state: dict) -> dict:
    """
    Search for real current job opportunities using Tavily.

    Inputs:
        state["target_role"]
        state["location"] or state["target_location"]
        state["search_retries"]

    Outputs:
        state["jobs"]
        state["search_queries"]
        state["job_scout_error"]
    """

    tavily_api_key = os.getenv("TAVILY_API_KEY")

    if not tavily_api_key:
        return {
            "jobs": [],
            "search_queries": [],
            "job_scout_error": "TAVILY_API_KEY is missing.",
        }

    target_role = str(
        state.get("target_role", "")
    ).strip()

    location = str(
        state.get("location")
        or state.get("target_location")
        or ""
    ).strip()

    search_retries = int(
        state.get("search_retries", 0)
    )

    if not target_role:
        return {
            "jobs": [],
            "search_queries": [],
            "job_scout_error": "Target role is missing.",
        }

    if not location:
        return {
            "jobs": [],
            "search_queries": [],
            "job_scout_error": "Target location is missing.",
        }

    search_queries = _build_search_queries(
        target_role=target_role,
        location=location,
        retry_count=search_retries,
    )

    client = TavilyClient(
        api_key=tavily_api_key
    )

    collected_jobs = []

    try:
        for query in search_queries:
            response = client.search(
                query=query,
                search_depth="advanced",
                max_results=MAX_RESULTS_PER_SEARCH,
                include_answer=False,
                include_raw_content=False,
            )

            results = response.get("results", [])

            for result in results:
                title = str(
                    result.get("title", "")
                ).strip()

                url = str(
                    result.get("url", "")
                ).strip()

                content = str(
                    result.get("content", "")
                ).strip()

                score = result.get("score")

                if not title:
                    continue

                if not _is_valid_url(url):
                    continue

                company = _extract_company_from_title(
                    title
                )

                collected_jobs.append(
                    {
                        "title": title,
                        "company": company,
                        "location": location,
                        "url": url,
                        "description": content,
                        "source": urlparse(url).netloc,
                        "search_score": score,
                        "search_query": query,
                        "retrieved_at": datetime.now(
                            timezone.utc
                        ).isoformat(),
                    }
                )

        jobs = _deduplicate_jobs(
            collected_jobs
        )

        return {
            "jobs": jobs,
            "search_queries": search_queries,
            "job_scout_error": None,
        }

    except Exception as error:
        return {
            "jobs": [],
            "search_queries": search_queries,
            "job_scout_error": (
                f"Job search failed: "
                f"{type(error).__name__}"
            ),
        }


# ==========================================
# 4. Local Test
# ==========================================

if __name__ == "__main__":
    test_state = {
        "target_role": "AI Engineer",
        "location": "Riyadh, Saudi Arabia",
        "search_retries": 0,
    }

    print("--- Running Job Scout Agent ---")

    output = job_scout_agent(test_state)

    print("\nSearch Queries:")
    for query in output["search_queries"]:
        print("-", query)

    print("\nJobs Found:")

    for index, job in enumerate(
        output["jobs"],
        start=1,
    ):
        print(f"\n{index}. {job['title']}")
        print("Company:", job["company"])
        print("Location:", job["location"])
        print("URL:", job["url"])
        print("Source:", job["source"])

    print("\nError:")
    print(output["job_scout_error"])