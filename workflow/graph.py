from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from workflow.analysis_nodes import (
    career_coach_node,
    gap_analyzer_node,
    requirements_node,
    route_gap_analysis,
    route_requirements_result,
)
from workflow.input_nodes import (
    input_guard_node,
    parallel_start_node,
    profile_analyzer_node,
    profile_ready_node,
    route_input_guard,
)
from workflow.job_nodes import (
    human_job_selection_node,
    job_scout_node,
    job_validation_node,
    jobs_branch_done_node,
    jobs_ready_node,
    limited_jobs_ready_node,
    refine_search_node,
    route_human_selection,
    route_job_validation,
    selected_job_enrichment_node,
)
from workflow.output_nodes import controlled_failure_node, final_report_node
from workflow.state import SkillGapState
from workflow.supervision_nodes import (
    career_coach_retry_node,
    profile_repair_node,
    profile_retry_node,
    requirements_retry_node,
    route_supervisor,
    supervisor_node,
)


def build_graph():

    workflow = StateGraph(
        SkillGapState
    )

    # --------------------------------------
    # Nodes
    # --------------------------------------

    workflow.add_node(
        "input_guard",
        input_guard_node,
    )

    workflow.add_node(
        "parallel_start",
        parallel_start_node,
    )

    workflow.add_node(
        "profile_analyzer",
        profile_analyzer_node,
    )

    workflow.add_node(
        "job_scout",
        job_scout_node,
    )

    workflow.add_node(
        "job_validation",
        job_validation_node,
    )

    workflow.add_node(
        "refine_search",
        refine_search_node,
    )

    workflow.add_node(
        "profile_ready",
        profile_ready_node,
    )

    workflow.add_node(
        "jobs_ready",
        jobs_ready_node,
    )

    workflow.add_node(
        "limited_jobs_ready",
        limited_jobs_ready_node,
    )

    workflow.add_node(
        "jobs_branch_done",
        jobs_branch_done_node,
    )

    workflow.add_node(
        "human_job_selection",
        human_job_selection_node,
    )

    workflow.add_node(
        "selected_job_enrichment",
        selected_job_enrichment_node,
    )

    workflow.add_node(
        "requirements",
        requirements_node,
    )

    workflow.add_node(
        "gap_analyzer",
        gap_analyzer_node,
    )

    workflow.add_node(
        "career_coach",
        career_coach_node,
    )

    workflow.add_node(
        "supervisor",
        supervisor_node,
    )

    workflow.add_node(
        "profile_retry",
        profile_retry_node,
    )

    workflow.add_node(
        "requirements_retry",
        requirements_retry_node,
    )

    workflow.add_node(
        "career_coach_retry",
        career_coach_retry_node,
    )

    workflow.add_node(
        "profile_repair",
        profile_repair_node,
    )

    workflow.add_node(
        "final_report",
        final_report_node,
    )

    workflow.add_node(
        "controlled_failure",
        controlled_failure_node,
    )

    # --------------------------------------
    # Start
    # --------------------------------------

    workflow.add_edge(
        START,
        "input_guard",
    )

    workflow.add_conditional_edges(
        "input_guard",
        route_input_guard,
        {
            "parallel_start":
                "parallel_start",

            "controlled_failure":
                "controlled_failure",
        },
    )

    # ======================================
    # PARALLEL FAN-OUT
    # ======================================

    workflow.add_edge(
        "parallel_start",
        "profile_analyzer",
    )

    workflow.add_edge(
        "parallel_start",
        "job_scout",
    )

    # --------------------------------------
    # Profile Branch
    # --------------------------------------

    workflow.add_edge(
        "profile_analyzer",
        "supervisor",
    )

    # --------------------------------------
    # Job Search Branch
    # --------------------------------------

    workflow.add_edge(
        "job_scout",
        "job_validation",
    )

    workflow.add_conditional_edges(
        "job_validation",
        route_job_validation,
        {
            "jobs_ready":
                "jobs_ready",

            "limited_jobs_ready":
                "limited_jobs_ready",

            "refine_search":
                "refine_search",

            "controlled_failure":
                "controlled_failure",
        },
    )

    workflow.add_edge(
        "refine_search",
        "job_scout",
    )

    workflow.add_edge(
        "jobs_ready",
        "jobs_branch_done",
    )

    workflow.add_edge(
        "limited_jobs_ready",
        "jobs_branch_done",
    )

    # --------------------------------------
    # Supervisor Routing
    # --------------------------------------

    workflow.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "profile_ready":
                "profile_ready",

            "gap_analyzer":
                "gap_analyzer",

            "final_report":
                "final_report",

            "profile_retry":
                "profile_retry",

            "requirements_retry":
                "requirements_retry",

            "career_coach_retry":
                "career_coach_retry",

            "controlled_failure":
                "controlled_failure",
        },
    )

    workflow.add_edge(
        "profile_retry",
        "profile_analyzer",
    )

    # ======================================
    # PARALLEL FAN-IN
    # ======================================
    #
    # The workflow waits for BOTH:
    # 1. approved candidate profile
    # 2. validated job results
    #
    # before Human-in-the-Loop starts.
    # ======================================

    workflow.add_edge(
        [
            "profile_ready",
            "jobs_branch_done",
        ],
        "human_job_selection",
    )

    # --------------------------------------
    # Human Selection
    # --------------------------------------

    workflow.add_conditional_edges(
        "human_job_selection",
        route_human_selection,
        {
            "human_job_selection":
                "human_job_selection",

            "selected_job_enrichment":
                "selected_job_enrichment",

            "controlled_failure":
                "controlled_failure",
        },
    )

    workflow.add_edge(
    "selected_job_enrichment",
    "requirements",
)

# --------------------------------------
# Requirements
# --------------------------------------

    workflow.add_conditional_edges(
    "requirements",
    route_requirements_result,
    {
        "supervisor": "supervisor",
        "retry": "requirements_retry",
        "failure": "controlled_failure",
    },
)

    workflow.add_edge(
    "requirements_retry",
    "requirements",
)

    # --------------------------------------
    # Gap Analyzer / Quality Check
    # --------------------------------------

    workflow.add_conditional_edges(
        "gap_analyzer",
        route_gap_analysis,
        {
            "career_coach":
                "career_coach",

            "requirements_retry":
                "requirements_retry",

            "profile_repair":
                "profile_repair",

            "controlled_failure":
                "controlled_failure",
        },
    )

    workflow.add_edge(
        "profile_repair",
        "profile_analyzer",
    )

    # --------------------------------------
    # Career Coach
    # --------------------------------------

    workflow.add_edge(
        "career_coach",
        "supervisor",
    )

    workflow.add_edge(
        "career_coach_retry",
        "career_coach",
    )

    # --------------------------------------
    # End
    # --------------------------------------

    workflow.add_edge(
        "final_report",
        END,
    )

    workflow.add_edge(
        "controlled_failure",
        END,
    )

    # --------------------------------------
    # Checkpointer required for interrupt()
    # --------------------------------------

    checkpointer = InMemorySaver()

    return workflow.compile(
        checkpointer=checkpointer
    )


graph = build_graph()
