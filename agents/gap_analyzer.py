import re
from typing import Any


SKILL_ALIASES = {
    "apis": "api",
    "rest api": "api",
    "rest apis": "api",
    "restful api": "api",
    "llms": "llm",
    "large language model": "llm",
    "large language models": "llm",
    "gen ai": "generative ai",
    "genai": "generative ai",
    "ml": "machine learning",
    "natural language processing": "nlp",
    "retrieval augmented generation": "rag",
    "python programming": "python",
    "scikit learn": "sklearn",
    "github": "git",
}


def normalize_skill(skill: Any) -> str:
    """Normalize skill names before deterministic comparison."""

    normalized = str(skill).casefold().strip()
    normalized = normalized.replace("_", " ")
    normalized = normalized.replace("-", " ")
    normalized = re.sub(r"[^\w+#.]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    return SKILL_ALIASES.get(normalized, normalized)


def unique_skills(skills: list[Any]) -> list[str]:
    """Remove duplicate and empty skills."""

    unique = []
    seen = set()

    for skill in skills:
        skill_text = str(skill).strip()
        normalized = normalize_skill(skill_text)

        if normalized and normalized not in seen:
            seen.add(normalized)
            unique.append(skill_text)

    return unique


def supported_skill_names(candidate_profile: dict) -> set[str]:
    """Find skills that have clear evidence in the candidate profile."""

    supported = set()

    evidence_map = candidate_profile.get("skill_evidence", {})

    if isinstance(evidence_map, dict):
        for skill, evidence in evidence_map.items():
            if isinstance(evidence, str) and evidence.strip():
                supported.add(normalize_skill(skill))

            elif isinstance(evidence, list) and evidence:
                supported.add(normalize_skill(skill))

    explicit_supported = candidate_profile.get(
        "evidence_supported_skills",
        [],
    )

    if isinstance(explicit_supported, list):
        for skill in explicit_supported:
            supported.add(normalize_skill(skill))

    return supported


def gap_analyzer(state: dict) -> dict:
    """
    Compare candidate skills with job requirements using Python.

    Inputs:
        state["candidate_profile"]
        state["job_requirements"]

    Outputs:
        state["gap_analysis"]
        state["gap_error"]
        state["gap_error_source"]
    """

    candidate_profile = state.get("candidate_profile")
    job_requirements = state.get("job_requirements")

    if not isinstance(candidate_profile, dict):
        return {
            "gap_analysis": {},
            "gap_error": "Candidate profile is missing or invalid.",
            "gap_error_source": "profile",
        }

    if not isinstance(job_requirements, dict):
        return {
            "gap_analysis": {},
            "gap_error": "Job requirements are missing or invalid.",
            "gap_error_source": "requirements",
        }

    candidate_skills = unique_skills(
        candidate_profile.get("skills", [])
        + candidate_profile.get("technical_skills", [])
    )

    required_skills = unique_skills(
        job_requirements.get("required_skills", [])
    )

    preferred_skills = unique_skills(
        job_requirements.get("preferred_skills", [])
    )

    if not candidate_skills:
        return {
            "gap_analysis": {},
            "gap_error": "Candidate profile contains no skills.",
            "gap_error_source": "profile",
        }

    if not required_skills:
        return {
            "gap_analysis": {},
            "gap_error": "Job requirements contain no required skills.",
            "gap_error_source": "requirements",
        }

    candidate_keys = {
        normalize_skill(skill) for skill in candidate_skills
    }

    matching_required = [
        skill
        for skill in required_skills
        if normalize_skill(skill) in candidate_keys
    ]

    missing_required = [
        skill
        for skill in required_skills
        if normalize_skill(skill) not in candidate_keys
    ]

    matching_preferred = [
        skill
        for skill in preferred_skills
        if normalize_skill(skill) in candidate_keys
    ]

    missing_preferred = [
        skill
        for skill in preferred_skills
        if normalize_skill(skill) not in candidate_keys
    ]

    supported_keys = supported_skill_names(candidate_profile)

    unsupported_skills = [
        skill
        for skill in candidate_skills
        if normalize_skill(skill) not in supported_keys
    ]

    unsupported_required = [
        skill
        for skill in matching_required
        if normalize_skill(skill) not in supported_keys
    ]

    skill_coverage = round(
        len(matching_required) / len(required_skills) * 100,
        1,
    )

    priority_gaps = []

    for skill in missing_required:
        priority_gaps.append(
            {
                "skill": skill,
                "priority": "high",
                "reason": "Required skill missing from the CV.",
            }
        )

    for skill in unsupported_required:
        priority_gaps.append(
            {
                "skill": skill,
                "priority": "medium",
                "reason": (
                    "Required skill is listed in the CV but lacks "
                    "clear evidence."
                ),
            }
        )

    for skill in missing_preferred:
        priority_gaps.append(
            {
                "skill": skill,
                "priority": "low",
                "reason": "Preferred skill missing from the CV.",
            }
        )

    gap_analysis = {
        "matching_skills": unique_skills(
            matching_required + matching_preferred
        ),
        "matching_required_skills": matching_required,
        "matching_preferred_skills": matching_preferred,
        "missing_skills": missing_required,
        "missing_required_skills": missing_required,
        "missing_preferred_skills": missing_preferred,
        "unsupported_skills": unsupported_skills,
        "priority_gaps": priority_gaps,
        "skill_coverage": skill_coverage,
        "coverage_details": {
            "matched_required": len(matching_required),
            "total_required": len(required_skills),
        },
    }

    return {
        "gap_analysis": gap_analysis,
        "gap_error": None,
        "gap_error_source": None,
    }