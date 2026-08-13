import os
from typing import TypedDict, Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

from agents.requirements_agent import requirements_agent
from agents.gap_analyzer import gap_analyzer
from agents.career_coach import CareerCoachAgent
from agents.supervisor import SupervisorAgent


load_dotenv()


# =========================
# Shared State
# =========================

class SkillGapState(TypedDict, total=False):
    # User input
    cv_text: str
    target_role: str
    location: str

    # Profile Analyzer output
    candidate_profile: dict

    # Job Scout output
    jobs: list[dict]
    search_retries: int
    search_error: str | None

    # Human-in-the-Loop
    selected_job_index: int
    selected_job: dict

    # Requirements Agent
    job_requirements: dict
    requirements_error: str | None

    # Gap Analyzer
    gap_analysis: dict
    gap_error: str | None
    gap_error_source: str | None

    # Career Coach
    recommendations: dict

    # Supervisor
    review_stage: str
    supervisor_decision: str
    supervisor_feedback: str

    # Retry counters
    profile_retry_count: int
    requirements_retry_count: int
    coach_retry_count: int

    # Final
    final_report: str
    error_message: str

    # Logs
    execution_logs: list[str]


# =========================
# Constants
# =========================

MAX_PROFILE_RETRIES = 1
MAX_REQUIREMENTS_RETRIES = 1
MAX_COACH_RETRIES = 1
MAX_SEARCH_RETRIES = 2


# =========================
# LLM
# =========================

def create_llm():
    api_key = os.getenv("OPENROUTER_API_KEY")
    model_name = os.getenv("OPENROUTER_MODEL")

    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is missing")

    if not model_name:
        raise ValueError("OPENROUTER_MODEL is missing")

    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.1,
    )


llm = create_llm()

career_coach_agent = CareerCoachAgent()
supervisor_agent = SupervisorAgent()


# =========================
# Helper
# =========================

def add_log(state: dict, message: str) -> list[str]:
    logs = list(state.get("execution_logs", []))
    logs.append(message)
    return logs


# =========================
# Requirements Node
# =========================

def requirements_node(state: SkillGapState) -> dict:
    result = requirements_agent(state, llm)

    return {
        **result,
        "review_stage": "requirements",
        "execution_logs": add_log(
            state,
            "[Requirements Agent] Job requirements analyzed",
        ),
    }


# =========================
# Gap Analyzer Node
# =========================

def gap_analyzer_node(state: SkillGapState) -> dict:
    result = gap_analyzer(state)

    return {
        **result,
        "execution_logs": add_log(
            state,
            "[Gap Analyzer] Skill gap analysis completed",
        ),
    }


# =========================
# Career Coach Node
# =========================

def career_coach_node(state: SkillGapState) -> dict:
    result = career_coach_agent.generate_recommendations(state)

    recommendations = (
        result.model_dump()
        if hasattr(result, "model_dump")
        else result.dict()
    )

    return {
        "recommendations": recommendations,
        "review_stage": "career_coach",
        "execution_logs": add_log(
            state,
            "[Career Coach] Recommendations generated",
        ),
    }


# =========================
# Supervisor Node
# =========================

def supervisor_node(state: SkillGapState) -> dict:
    result = supervisor_agent.review(state)

    return {
        "supervisor_decision": result.decision,
        "supervisor_feedback": result.feedback,
        "execution_logs": add_log(
            state,
            (
                "[Supervisor] "
                f"{state.get('review_stage', 'unknown')} "
                f"review → {result.decision}"
            ),
        ),
    }


# =========================
# Supervisor Routing
# =========================

def route_supervisor(state: SkillGapState) -> str:
    decision = state.get("supervisor_decision")

    if decision == "approve":
        review_stage = state.get("review_stage")

        if review_stage == "requirements":
            return "gap_analyzer"

        if review_stage == "career_coach":
            return "final_report"

        if review_stage == "profile":
            return "job_scout"

        return "controlled_failure"

    if decision == "retry_profile":
        retry_count = state.get("profile_retry_count", 0)

        if retry_count < MAX_PROFILE_RETRIES:
            return "profile_analyzer"

        return "controlled_failure"

    if decision == "retry_requirements":
        retry_count = state.get("requirements_retry_count", 0)

        if retry_count < MAX_REQUIREMENTS_RETRIES:
            return "requirements_retry"

        return "controlled_failure"

    if decision == "retry_career_coach":
        retry_count = state.get("coach_retry_count", 0)

        if retry_count < MAX_COACH_RETRIES:
            return "career_coach_retry"

        return "controlled_failure"

    return "controlled_failure"


# =========================
# Retry Nodes
# =========================

def requirements_retry_node(state: SkillGapState) -> dict:
    return {
        "requirements_retry_count":
            state.get("requirements_retry_count", 0) + 1,
    }


def career_coach_retry_node(state: SkillGapState) -> dict:
    return {
        "coach_retry_count":
            state.get("coach_retry_count", 0) + 1,
    }


# =========================
# Gap Quality Routing
# =========================

def route_gap_analysis(state: SkillGapState) -> str:
    error = state.get("gap_error")
    source = state.get("gap_error_source")

    if not error:
        return "career_coach"

    if source == "requirements":
        return "requirements"

    if source == "profile":
        return "profile_analyzer"

    return "controlled_failure"


# =========================
# Final Report
# =========================

def final_report_node(state: SkillGapState) -> dict:
    gap = state.get("gap_analysis", {})
    recommendations = state.get("recommendations", {})
    selected_job = state.get("selected_job", {})

    title = selected_job.get("title", "Not listed")
    company = selected_job.get("company", "Not listed")

    report = f"""
SKILLGAP AI
Career Opportunity Analysis

Target Role:
{state.get("target_role", "")}

Location:
{state.get("location", "")}

Selected Opportunity:
{title} — {company}

---------------------------------

STRONG MATCHES

{gap.get("matching_skills", [])}

---------------------------------

MISSING REQUIRED SKILLS

{gap.get("missing_required_skills", [])}

---------------------------------

SKILL COVERAGE

{gap.get("skill_coverage", 0)}%

---------------------------------

CAREER COACH RECOMMENDATION

{recommendations}
"""

    return {
        "final_report": report.strip(),
        "execution_logs": add_log(
            state,
            "[Final Report] Report generated",
        ),
    }


# =========================
# Controlled Failure
# =========================

def controlled_failure_node(state: SkillGapState) -> dict:
    message = (
        state.get("gap_error")
        or state.get("requirements_error")
        or state.get("supervisor_feedback")
        or "Workflow stopped because the analysis could not be validated."
    )

    return {
        "error_message": message,
        "execution_logs": add_log(
            state,
            f"[Controlled Failure] {message}",
        ),
    }


# =========================
# Graph Builder
# =========================

def build_graph():
    workflow = StateGraph(SkillGapState)

    # Current implemented nodes
    workflow.add_node("requirements", requirements_node)
    workflow.add_node("gap_analyzer", gap_analyzer_node)
    workflow.add_node("career_coach", career_coach_node)
    workflow.add_node("supervisor", supervisor_node)

    workflow.add_node(
        "requirements_retry",
        requirements_retry_node,
    )

    workflow.add_node(
        "career_coach_retry",
        career_coach_retry_node,
    )

    workflow.add_node("final_report", final_report_node)
    workflow.add_node(
        "controlled_failure",
        controlled_failure_node,
    )

    # Temporary starting point until Agents 1 & 2 are added
    workflow.add_edge(
        START,
        "requirements",
    )

    workflow.add_edge(
        "requirements",
        "supervisor",
    )

    workflow.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "gap_analyzer": "gap_analyzer",
            "career_coach": "career_coach",
            "final_report": "final_report",

            "requirements_retry": "requirements_retry",
            "career_coach_retry": "career_coach_retry",

            "controlled_failure": "controlled_failure",

            # These will work after Agents 1 & 2 are added
            # "profile_analyzer": "profile_analyzer",
            # "job_scout": "job_scout",
        },
    )

    workflow.add_edge(
        "requirements_retry",
        "requirements",
    )

    workflow.add_conditional_edges(
        "gap_analyzer",
        route_gap_analysis,
        {
            "career_coach": "career_coach",
            "requirements": "requirements",
            "controlled_failure": "controlled_failure",

            # Add after Profile Analyzer exists:
            # "profile_analyzer": "profile_analyzer",
        },
    )

    workflow.add_edge(
        "career_coach",
        "supervisor",
    )

    workflow.add_edge(
        "career_coach_retry",
        "career_coach",
    )

    workflow.add_edge(
        "final_report",
        END,
    )

    workflow.add_edge(
        "controlled_failure",
        END,
    )

    return workflow.compile()


graph = build_graph()