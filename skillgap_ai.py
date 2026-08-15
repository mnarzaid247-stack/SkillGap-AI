"""
Public entry point for SkillGap AI.

The workflow implementation lives in the workflow package.
Keeping this module small preserves existing imports such as:

    from skillgap_ai import graph
"""

from workflow.graph import build_graph, graph


__all__ = ["build_graph", "graph"]


if __name__ == "__main__":
    from cli import run_cli

    run_cli()
