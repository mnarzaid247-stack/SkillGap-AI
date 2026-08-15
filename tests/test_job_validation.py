from workflow.job_search_nodes import (
    _job_is_explicitly_closed,
    _job_looks_relevant,
    _valid_job_url,
    job_validation_node,
)


def make_job(
    title="Data Analyst",
    url="https://example.com/jobs/1",
    description="Python SQL Power BI data analysis",
):
    return {
        "title": title,
        "company": "Example Company",
        "location": "Riyadh",
        "description": description,
        "url": url,
        "source": "Example",
    }


def test_valid_job_url_accepts_https():
    assert _valid_job_url(
        "https://example.com/jobs/1"
    )


def test_valid_job_url_accepts_http():
    assert _valid_job_url(
        "http://example.com/jobs/1"
    )


def test_invalid_job_url_is_rejected():
    assert not _valid_job_url(
        "not-a-valid-url"
    )


def test_empty_job_url_is_rejected():
    assert not _valid_job_url("")


def test_relevant_job_is_detected():
    job = make_job(
        title="Data Analyst",
    )

    assert _job_looks_relevant(
        job,
        "Data Analyst",
    )


def test_irrelevant_job_is_rejected():
    job = make_job(
        title="Graphic Designer",
        description=(
            "Adobe Illustrator Photoshop "
            "branding and visual design"
        ),
    )

    assert not _job_looks_relevant(
        job,
        "Data Analyst",
    )


def test_related_ai_job_is_accepted():
    job = make_job(
        title="Machine Learning Engineer",
        description=(
            "Build machine learning models "
            "and AI applications"
        ),
    )

    assert _job_looks_relevant(
        job,
        "AI Engineer",
    )


def test_closed_job_is_detected():
    job = make_job(
        description=(
            "This position is closed and "
            "no longer accepting applications."
        ),
    )

    assert _job_is_explicitly_closed(job)


def test_open_job_is_not_marked_closed():
    job = make_job(
        description=(
            "Applications are currently open."
        ),
    )

    assert not _job_is_explicitly_closed(job)


def test_job_validation_keeps_valid_job():
    state = {
        "target_role": "Data Analyst",
        "jobs": [
            make_job(),
        ],
    }

    result = job_validation_node(state)

    assert result["valid_job_count"] == 1
    assert len(result["jobs"]) == 1
    assert result["job_validation_error"] is None


def test_job_validation_rejects_missing_title():
    job = make_job()
    job["title"] = ""

    state = {
        "target_role": "Data Analyst",
        "jobs": [job],
    }

    result = job_validation_node(state)

    assert result["valid_job_count"] == 0
    assert result["jobs"] == []


def test_job_validation_rejects_invalid_url():
    state = {
        "target_role": "Data Analyst",
        "jobs": [
            make_job(
                url="invalid-url"
            ),
        ],
    }

    result = job_validation_node(state)

    assert result["valid_job_count"] == 0


def test_job_validation_rejects_closed_job():
    state = {
        "target_role": "Data Analyst",
        "jobs": [
            make_job(
                description=(
                    "Data Analyst role. "
                    "Applications closed."
                )
            ),
        ],
    }

    result = job_validation_node(state)

    assert result["valid_job_count"] == 0


def test_job_validation_removes_duplicate_urls():
    job1 = make_job()
    job2 = make_job(
        title="Senior Data Analyst",
    )

    state = {
        "target_role": "Data Analyst",
        "jobs": [
            job1,
            job2,
        ],
    }

    result = job_validation_node(state)

    assert result["valid_job_count"] == 1


def test_job_validation_rejects_non_dict_items():
    state = {
        "target_role": "Data Analyst",
        "jobs": [
            "invalid job",
            None,
            make_job(),
        ],
    }

    result = job_validation_node(state)

    assert result["valid_job_count"] == 1