from workflow.runtime import _live_log
from workflow.state import SkillGapState


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
