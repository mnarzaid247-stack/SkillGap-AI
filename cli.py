import uuid

from langgraph.types import Command

from workflow.graph import graph


def _print_jobs_from_interrupt(
    interrupt_value,
):
    """
    Print the job list sent by the
    Human-in-the-Loop node.
    """

    if not isinstance(
        interrupt_value,
        dict,
    ):
        return

    jobs = interrupt_value.get(
        "jobs",
        [],
    )

    print(
        "\n=== CURRENT OPPORTUNITIES ==="
    )

    if interrupt_value.get(
        "limited_results"
    ):
        print(
            "\nWarning: Only a limited number "
            "of valid jobs were found.\n"
        )

    for job in jobs:
        print(
            f"\n{job['number']}. "
            f"{job['title']}"
        )

        print(
            "   Company:",
            job.get(
                "company"
            ) or "Not listed",
        )

        print(
            "   Location:",
            job.get(
                "location"
            ) or "Not listed",
        )

        print(
            "   Source:",
            job.get(
                "source"
            ) or "Not listed",
        )


def run_cli():

    print(
        "\n================================"
    )
    print(
        "           SKILLGAP AI"
    )
    print(
        "================================\n"
    )

    cv_path = input(
        "CV text file path: "
    ).strip()

    try:
        with open(
            cv_path,
            "r",
            encoding="utf-8",
        ) as file:
            cv_text = file.read()

    except Exception as error:
        print(
            "\nCould not read CV file:",
            type(error).__name__,
        )
        return

    target_role = input(
        "Target role: "
    ).strip()

    location = input(
        "Location: "
    ).strip()

    initial_state = {
        "cv_text":
            cv_text,

        "target_role":
            target_role,

        "location":
            location,

        "execution_logs":
            [],
    }

    thread_id = str(
        uuid.uuid4()
    )

    config = {
        "configurable": {
            "thread_id":
                thread_id,
        }
    }

    result = graph.invoke(
        initial_state,
        config=config,
    )

    # --------------------------------------
    # Human-in-the-Loop resume cycle
    # --------------------------------------

    while "__interrupt__" in result:

        interrupts = result[
            "__interrupt__"
        ]

        if not interrupts:
            break

        current_interrupt = (
            interrupts[0]
        )

        interrupt_value = getattr(
            current_interrupt,
            "value",
            current_interrupt,
        )

        _print_jobs_from_interrupt(
            interrupt_value
        )

        selection = input(
            "\nSelect job number: "
        ).strip()

        result = graph.invoke(
            Command(
                resume=selection
            ),
            config=config,
        )

    # --------------------------------------
    # Output
    # --------------------------------------

    if result.get(
        "error_message"
    ):
        print(
            "\n=== WORKFLOW ERROR ===\n"
        )

        print(
            result[
                "error_message"
            ]
        )

    elif result.get(
        "final_report"
    ):
        print(
            "\n\n"
            + result[
                "final_report"
            ]
        )

    print(
        "\n\n=== EXECUTION LOGS ==="
    )

    for log in result.get(
        "execution_logs",
        [],
    ):
        print(
            log
        )
