import re
from typing import Any


# ==========================================
# 1. Skill Aliases
# ==========================================

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


# ==========================================
# 2. Helper Functions
# ==========================================

def normalize_skill(skill: Any) -> str:
    """Normalize skill names before deterministic comparison."""

    normalized = str(skill).casefold().strip()
    normalized = normalized.replace("_", " ")
    normalized = normalized.replace("-", " ")
    normalized = re.sub(r"[^\w+#.]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    return SKILL_ALIASES.get(normalized, normalized)


def unique_skills(skills: list[Any]) -> list[str]:
    """Remove duplicate and empty skills while preserving display text."""

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
    """
    Find skills that have clear evidence in the candidate profile.

    Evidence comes from:
        candidate_profile["skill_evidence"]
    """

    supported = set()

    evidence_map = candidate_profile.get(
        "skill_evidence",
        {},
    )

    if isinstance(evidence_map, dict):
        for skill, evidence in evidence_map.items():

            if isinstance(evidence, str) and evidence.strip():
                supported.add(
                    normalize_skill(skill)
                )

            elif isinstance(evidence, list) and evidence:
                supported.add(
                    normalize_skill(skill)
                )

    return supported


def build_evidence_gaps(
    candidate_skills: list[str],
    supported_keys: set[str],
) -> list[dict]:
    """
    Create explicit evidence-gap records.

    A skill can exist in the CV while still lacking
    clear project or experience evidence.
    """

    evidence_gaps = []

    for skill in candidate_skills:
        if normalize_skill(skill) not in supported_keys:
            evidence_gaps.append(
                {
                    "skill": skill,
                    "reason": (
                        "Skill is listed in the CV but lacks "
                        "clear project, experience, education, "
                        "or practical implementation evidence."
                    ),
                }
            )

    return evidence_gaps


# ==========================================
# 3. Gap Analyzer
# ==========================================

def gap_analyzer(state: dict) -> dict:
    """
    Compare the candidate profile with selected job requirements
    using deterministic Python logic.

    Inputs:
        state["candidate_profile"]
        state["job_requirements"]

    Outputs:
        state["gap_analysis"]
        state["gap_error"]
        state["gap_error_source"]
    """

    candidate_profile = state.get(
        "candidate_profile"
    )

    job_requirements = state.get(
        "job_requirements"
    )

    # ------------------------------------------
    # Validate profile
    # ------------------------------------------

    if not isinstance(candidate_profile, dict):
        return {
            "gap_analysis": {},
            "gap_error": (
                "Candidate profile is missing or invalid."
            ),
            "gap_error_source": "profile",
        }

    # ------------------------------------------
    # Validate requirements
    # ------------------------------------------

    if not isinstance(job_requirements, dict):
        return {
            "gap_analysis": {},
            "gap_error": (
                "Job requirements are missing or invalid."
            ),
            "gap_error_source": "requirements",
        }

    # ------------------------------------------
    # Read skills
    # ------------------------------------------

    candidate_skills = unique_skills(
        candidate_profile.get("skills", [])
    )

    required_skills = unique_skills(
        job_requirements.get(
            "required_skills",
            [],
        )
    )

    preferred_skills = unique_skills(
        job_requirements.get(
            "preferred_skills",
            [],
        )
    )

    # ------------------------------------------
    # Basic quality checks
    # ------------------------------------------

    if not candidate_skills:
        return {
            "gap_analysis": {},
            "gap_error": (
                "Candidate profile contains no skills."
            ),
            "gap_error_source": "profile",
        }

    if not required_skills:
        return {
            "gap_analysis": {},
            "gap_error": (
                "Job requirements contain no required skills."
            ),
            "gap_error_source": "requirements",
        }

    # ------------------------------------------
    # Normalize candidate skills
    # ------------------------------------------

    candidate_keys = {
        normalize_skill(skill)
        for skill in candidate_skills
    }

    # ------------------------------------------
    # Required skill comparison
    # ------------------------------------------

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

    # ------------------------------------------
    # Preferred skill comparison
    # ------------------------------------------

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

    # ------------------------------------------
    # Evidence analysis
    # ------------------------------------------

    supported_keys = supported_skill_names(
        candidate_profile
    )

    evidence_gaps = build_evidence_gaps(
        candidate_skills,
        supported_keys,
    )

    unsupported_required = [
        skill
        for skill in matching_required
        if normalize_skill(skill) not in supported_keys
    ]

    # ------------------------------------------
    # Deterministic Skill Coverage
    # ------------------------------------------

    coverage_numerator = len(
        matching_required
    )

    coverage_denominator = len(
        required_skills
    )

    skill_coverage = round(
        (
            coverage_numerator
            / coverage_denominator
        )
        * 100,
        1,
    )

    # ------------------------------------------
    # Priority Gaps
    # ------------------------------------------

    priority_gaps = []

    # Missing required skills = highest priority
    for skill in missing_required:
        priority_gaps.append(
            {
                "skill": skill,
                "priority": "HIGH",
                "reason": (
                    "Required skill is missing from the CV."
                ),
            }
        )

    # Present but unsupported required skills
    for skill in unsupported_required:
        priority_gaps.append(
            {
                "skill": skill,
                "priority": "MEDIUM",
                "reason": (
                    "Required skill is listed in the CV "
                    "but lacks clear evidence."
                ),
            }
        )

    # Missing preferred skills = lower priority
    for skill in missing_preferred:
        priority_gaps.append(
            {
                "skill": skill,
                "priority": "LOW",
                "reason": (
                    "Preferred skill is missing from the CV."
                ),
            }
        )

    # ------------------------------------------
    # Final Gap Analysis
    # ------------------------------------------

    gap_analysis = {
        "matching_skills": unique_skills(
            matching_required
            + matching_preferred
        ),

        "matching_required_skills":
            matching_required,

        "matching_preferred_skills":
            matching_preferred,

        "missing_skills":
            missing_required,

        "missing_required_skills":
            missing_required,

        "missing_preferred_skills":
            missing_preferred,

        "evidence_gaps":
            evidence_gaps,

        "unsupported_required_skills":
            unsupported_required,

        "priority_gaps":
            priority_gaps,

        "skill_coverage":
            skill_coverage,

        "coverage_details": {
            "matched_required":
                coverage_numerator,

            "total_required":
                coverage_denominator,
        },
    }

    return {
        "gap_analysis": gap_analysis,
        "gap_error": None,
        "gap_error_source": None,
    }


# ==========================================
# 4. Local Test
# ==========================================

if __name__ == "__main__":
    test_state = {
        "candidate_profile": {
            "skills": [
                "Python",
                "RAG",
                "Git",
                "Prompt Engineering",
            ],
            "skill_evidence": {
                "Python": (
                    "Used in AI projects."
                ),
                "RAG": (
                    "Built a RAG workflow."
                ),
                "Git": (
                    "Used for version control."
                ),
            },
        },

        "job_requirements": {
            "required_skills": [
                "Python",
                "RAG",
                "SQL",
                "Docker",
            ],
            "preferred_skills": [
                "AWS",
                "Git",
            ],
        },
    }

    print("--- Running Gap Analyzer ---")

    output = gap_analyzer(test_state)

    print("\nGap Analysis:")
    print(output["gap_analysis"])

    print("\nGap Error:")
    print(output["gap_error"])

    print("\nGap Error Source:")
    print(output["gap_error_source"])