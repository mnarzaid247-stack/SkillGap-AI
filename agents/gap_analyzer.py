from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field


# ==========================================
# 1. Structured Match Schema
# ==========================================

class SkillMatch(BaseModel):
    requirement: str
    matched: bool
    evidence: list[str] = Field(
        default_factory=list
    )
    reason: str = ""


class GapMatchResult(BaseModel):
    required_matches: list[SkillMatch] = Field(
        default_factory=list
    )
    preferred_matches: list[SkillMatch] = Field(
        default_factory=list
    )


# ==========================================
# 2. Helpers
# ==========================================

def _model_to_dict(
    model: BaseModel,
) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()

    return model.dict()


def _clean_string_list(
    value: Any,
) -> list[str]:
    if not isinstance(
        value,
        list,
    ):
        return []

    cleaned = []
    seen = set()

    for item in value:
        text = str(
            item
        ).strip()

        if not text:
            continue

        key = text.casefold()

        if key in seen:
            continue

        seen.add(
            key
        )

        cleaned.append(
            text
        )

    return cleaned


def _unique_requirements(
    requirements: Any,
) -> list[str]:
    return _clean_string_list(
        requirements
    )


def _candidate_evidence_lines(
    candidate_profile: dict,
) -> list[str]:
    """
    Build grounded evidence lines from the candidate profile.

    These are the only facts the LLM is allowed to use
    when judging skill matches.
    """

    lines = []

    skills = _clean_string_list(
        candidate_profile.get(
            "skills",
            [],
        )
    )

    if skills:
        lines.append(
            "Explicit skills: "
            + ", ".join(skills)
        )

    evidence_map = candidate_profile.get(
        "skill_evidence",
        {},
    )

    if isinstance(
        evidence_map,
        dict,
    ):
        for skill, evidence in evidence_map.items():
            skill_name = str(
                skill
            ).strip()

            if not skill_name:
                continue

            if isinstance(
                evidence,
                str,
            ):
                evidence_text = (
                    evidence.strip()
                )

                if evidence_text:
                    lines.append(
                        f"{skill_name}: {evidence_text}"
                    )

            elif isinstance(
                evidence,
                list,
            ):
                items = _clean_string_list(
                    evidence
                )

                if items:
                    lines.append(
                        f"{skill_name}: "
                        + " | ".join(items)
                    )

    projects = candidate_profile.get(
        "projects",
        []
    )

    if isinstance(
        projects,
        list,
    ):
        for project in projects:
            if isinstance(
                project,
                str,
            ):
                text = project.strip()

                if text:
                    lines.append(
                        f"Project: {text}"
                    )

            elif isinstance(
                project,
                dict,
            ):
                title = str(
                    project.get(
                        "title",
                        "",
                    )
                ).strip()

                description = str(
                    project.get(
                        "description",
                        "",
                    )
                ).strip()

                if title or description:
                    lines.append(
                        "Project: "
                        + " — ".join(
                            part
                            for part in [
                                title,
                                description,
                            ]
                            if part
                        )
                    )

    experience = candidate_profile.get(
        "experience",
        []
    )

    if isinstance(
        experience,
        list,
    ):
        for item in experience:
            if isinstance(
                item,
                str,
            ):
                text = item.strip()

                if text:
                    lines.append(
                        f"Experience: {text}"
                    )

            elif isinstance(
                item,
                dict,
            ):
                parts = [
                    str(
                        item.get(
                            key,
                            "",
                        )
                    ).strip()
                    for key in (
                        "title",
                        "company",
                        "description",
                    )
                ]

                parts = [
                    part
                    for part in parts
                    if part
                ]

                if parts:
                    lines.append(
                        "Experience: "
                        + " — ".join(parts)
                    )

    education = candidate_profile.get(
        "education",
        []
    )

    if isinstance(
        education,
        list,
    ):
        for item in education:
            if isinstance(
                item,
                str,
            ):
                text = item.strip()

                if text:
                    lines.append(
                        f"Education: {text}"
                    )

            elif isinstance(
                item,
                dict,
            ):
                parts = [
                    str(
                        item.get(
                            key,
                            "",
                        )
                    ).strip()
                    for key in (
                        "degree",
                        "field",
                        "institution",
                    )
                ]

                parts = [
                    part
                    for part in parts
                    if part
                ]

                if parts:
                    lines.append(
                        "Education: "
                        + " — ".join(parts)
                    )

    return lines


def _match_map(
    matches: list[SkillMatch],
) -> dict[str, SkillMatch]:
    """
    Index model matches by exact requirement text.

    The exact requirement text is preserved so downstream
    output always references the original extracted skill.
    """

    result = {}

    for match in matches:
        requirement = str(
            match.requirement
        ).strip()

        if not requirement:
            continue

        result[
            requirement.casefold()
        ] = match

    return result


def _align_matches(
    requirements: list[str],
    model_matches: list[SkillMatch],
) -> list[dict]:
    """
    Align LLM output with the original requirements.

    This prevents the model from:
    - adding new requirements
    - silently dropping requirements
    - changing denominator size
    """

    indexed = _match_map(
        model_matches
    )

    aligned = []

    for requirement in requirements:
        match = indexed.get(
            requirement.casefold()
        )

        if match is None:
            aligned.append(
                {
                    "requirement":
                        requirement,
                    "matched":
                        False,
                    "evidence":
                        [],
                    "reason": (
                        "No validated match was returned "
                        "for this requirement."
                    ),
                }
            )
            continue

        evidence = _clean_string_list(
            match.evidence
        )

        aligned.append(
            {
                "requirement":
                    requirement,
                "matched":
                    bool(
                        match.matched
                    ),
                "evidence":
                    evidence,
                "reason":
                    str(
                        match.reason
                    ).strip(),
            }
        )

    return aligned


def _build_evidence_gaps(
    candidate_profile: dict,
) -> list[dict]:
    """
    Preserve the original evidence-gap idea.

    A CV-listed skill without explicit supporting evidence
    is surfaced separately, but it does not automatically
    reduce technical skill coverage.
    """

    skills = _clean_string_list(
        candidate_profile.get(
            "skills",
            [],
        )
    )

    evidence_map = candidate_profile.get(
        "skill_evidence",
        {},
    )

    supported = set()

    if isinstance(
        evidence_map,
        dict,
    ):
        for skill, evidence in evidence_map.items():
            has_evidence = (
                isinstance(
                    evidence,
                    str,
                )
                and bool(
                    evidence.strip()
                )
            ) or (
                isinstance(
                    evidence,
                    list,
                )
                and bool(
                    evidence
                )
            )

            if has_evidence:
                supported.add(
                    str(
                        skill
                    ).strip().casefold()
                )

    gaps = []

    for skill in skills:
        if (
            skill.casefold()
            not in supported
        ):
            gaps.append(
                {
                    "skill":
                        skill,
                    "reason": (
                        "Skill is listed in the CV but lacks "
                        "clear supporting evidence in the "
                        "extracted candidate profile."
                    ),
                }
            )

    return gaps


# ==========================================
# 3. Gap Analyzer Agent
# ==========================================

def gap_analyzer(
    state: dict,
    llm: BaseChatModel,
) -> dict:
    """
    Compare candidate evidence with job requirements.

    The LLM performs semantic evidence-based matching.
    Python performs deterministic validation, alignment,
    coverage calculation, and priority-gap construction.
    """

    candidate_profile = state.get(
        "candidate_profile"
    )

    job_requirements = state.get(
        "job_requirements"
    )

    # ------------------------------------------
    # Validate inputs
    # ------------------------------------------

    if not isinstance(
        candidate_profile,
        dict,
    ):
        return {
            "gap_analysis": {},
            "gap_error": (
                "Candidate profile is missing or invalid."
            ),
            "gap_error_source": "profile",
        }

    if not isinstance(
        job_requirements,
        dict,
    ):
        return {
            "gap_analysis": {},
            "gap_error": (
                "Job requirements are missing or invalid."
            ),
            "gap_error_source": "requirements",
        }

    required_skills = (
        _unique_requirements(
            job_requirements.get(
                "required_skills",
                [],
            )
        )
    )

    preferred_skills = (
        _unique_requirements(
            job_requirements.get(
                "preferred_skills",
                [],
            )
        )
    )

    candidate_skills = (
        _clean_string_list(
            candidate_profile.get(
                "skills",
                [],
            )
        )
    )

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

    evidence_lines = (
        _candidate_evidence_lines(
            candidate_profile
        )
    )

    if not evidence_lines:
        return {
            "gap_analysis": {},
            "gap_error": (
                "Candidate profile contains no usable evidence."
            ),
            "gap_error_source": "profile",
        }

    # ------------------------------------------
    # Prompt
    # ------------------------------------------

    system_message = SystemMessage(
        content=(
            "You are the Gap Analyzer Agent in SkillGap AI. "
            "Your task is to judge whether each job requirement "
            "is supported by explicit evidence in the candidate profile. "
            "\n\n"
            "You perform semantic matching, but you must remain grounded. "
            "A requirement may match even when wording differs, as long "
            "as the candidate evidence clearly demonstrates the same or "
            "a sufficiently specific capability. "
            "\n\n"
            "Do NOT infer a skill only from a job title, degree title, "
            "or a vaguely related neighboring skill. "
            "\n\n"
            "For every matched requirement, cite the exact candidate "
            "evidence that supports the decision. "
            "\n\n"
            "Do not calculate percentages. "
            "Do not add, remove, rename, or rewrite requirements."
        )
    )

    candidate_evidence = "\n".join(
        f"- {line}"
        for line in evidence_lines
    )

    required_text = "\n".join(
        f"- {skill}"
        for skill in required_skills
    )

    preferred_text = (
        "\n".join(
            f"- {skill}"
            for skill in preferred_skills
        )
        if preferred_skills
        else "- None"
    )

    human_message = HumanMessage(
        content=f"""
Candidate evidence:
<candidate_profile>
{candidate_evidence}
</candidate_profile>

Required technical skills:
<required_skills>
{required_text}
</required_skills>

Preferred technical skills:
<preferred_skills>
{preferred_text}
</preferred_skills>

Matching rules:

1. Evaluate EVERY required skill exactly once.

2. Evaluate EVERY preferred skill exactly once.

3. Keep the requirement text EXACTLY as provided.

4. matched=true only when explicit candidate evidence
   supports the requirement.

5. Semantic equivalence is allowed.

   Example:
   Requirement: Statistical Analysis
   Candidate evidence: Statistical Modeling, Regression Analysis,
   Time Series Analysis
   -> This can be a valid match when the evidence clearly shows
   practical statistical analysis capability.

6. A more specific demonstrated capability may satisfy a broader
   requirement.

7. Do NOT use unsupported assumptions.

   Bad examples:
   - Statistics degree -> automatically knows every statistical tool
   - Data Analyst -> automatically knows SQL
   - Python -> automatically knows Machine Learning
   - Power BI -> automatically knows Tableau

8. For matched=true:
   evidence must contain one or more short pieces of candidate
   evidence from the provided profile.

9. For matched=false:
   evidence must be an empty list.

10. reason must briefly explain the decision.

11. Do not calculate Skill Coverage.

12. Do not create new requirements.

Return only structured output matching the provided schema.
"""
    )

    # ------------------------------------------
    # Semantic matching
    # ------------------------------------------

    try:
        structured_llm = (
            llm.with_structured_output(
                GapMatchResult
            )
        )

        model_result = (
            structured_llm.invoke(
                [
                    system_message,
                    human_message,
                ]
            )
        )

        if not isinstance(
            model_result,
            GapMatchResult,
        ):
            raise ValueError(
                "Gap Analyzer did not return "
                "the expected structured output."
            )

    except Exception as error:
        return {
            "gap_analysis": {},
            "gap_error": (
                "Semantic gap matching failed: "
                f"{type(error).__name__}"
            ),
            "gap_error_source": "gap_analyzer",
        }

    # ------------------------------------------
    # Deterministic alignment
    # ------------------------------------------

    required_results = (
        _align_matches(
            required_skills,
            model_result.required_matches,
        )
    )

    preferred_results = (
        _align_matches(
            preferred_skills,
            model_result.preferred_matches,
        )
    )

    # ------------------------------------------
    # Deterministic result sets
    # ------------------------------------------

    matching_required = [
        item["requirement"]
        for item in required_results
        if item["matched"]
    ]

    missing_required = [
        item["requirement"]
        for item in required_results
        if not item["matched"]
    ]

    matching_preferred = [
        item["requirement"]
        for item in preferred_results
        if item["matched"]
    ]

    missing_preferred = [
        item["requirement"]
        for item in preferred_results
        if not item["matched"]
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
    # Evidence gaps
    # ------------------------------------------

    evidence_gaps = (
        _build_evidence_gaps(
            candidate_profile
        )
    )

    # Requirements that matched semantically but
    # have no returned supporting evidence are not
    # trusted as fully evidenced.
    unsupported_required = [
        item["requirement"]
        for item in required_results
        if (
            item["matched"]
            and not item["evidence"]
        )
    ]

    # ------------------------------------------
    # Priority gaps
    # ------------------------------------------

    priority_gaps = []

    for skill in missing_required:
        match_record = next(
            (
                item
                for item in required_results
                if item["requirement"] == skill
            ),
            {},
        )

        priority_gaps.append(
            {
                "skill": skill,
                "priority": "HIGH",
                "reason": (
                    match_record.get(
                        "reason"
                    )
                    or (
                        "Required technical skill is not "
                        "supported by the candidate profile."
                    )
                ),
            }
        )

    for skill in unsupported_required:
        priority_gaps.append(
            {
                "skill": skill,
                "priority": "MEDIUM",
                "reason": (
                    "The semantic matcher marked this requirement "
                    "as present but did not provide supporting evidence."
                ),
            }
        )

    for skill in missing_preferred:
        match_record = next(
            (
                item
                for item in preferred_results
                if item["requirement"] == skill
            ),
            {},
        )

        priority_gaps.append(
            {
                "skill": skill,
                "priority": "LOW",
                "reason": (
                    match_record.get(
                        "reason"
                    )
                    or (
                        "Preferred technical skill is not "
                        "supported by the candidate profile."
                    )
                ),
            }
        )

    # ------------------------------------------
    # Final Gap Analysis
    # ------------------------------------------

    gap_analysis = {
        "matching_skills":
            _unique_requirements(
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

        "required_match_details":
            required_results,

        "preferred_match_details":
            preferred_results,

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
        "gap_analysis":
            gap_analysis,

        "gap_error":
            None,

        "gap_error_source":
            None,
    }