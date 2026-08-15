import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langgraph.types import Command
from pydantic import BaseModel

from api_helpers import (
    extract_pdf_text,
    get_interrupt_payload,
    read_and_validate_pdf,
)
from skillgap_ai import graph


app = FastAPI(
    title="SkillGap AI API",
    version="1.0.0",
)


# ==========================================
# Frontend
# ==========================================

FRONTEND_DIR = Path(__file__).parent / "frontend"

app.mount(
    "/frontend",
    StaticFiles(directory=FRONTEND_DIR),
    name="frontend",
)


@app.get("/")
def home():
    return FileResponse(
        FRONTEND_DIR / "index.html"
    )


# ==========================================
# Request Schemas
# ==========================================

class JobSelectionRequest(BaseModel):
    thread_id: str
    selected_job_index: int


# ==========================================
# Health Check
# ==========================================

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "service": "SkillGap AI",
    }


# ==========================================
# Start Analysis
# ==========================================

@app.post("/api/start")
async def start_analysis(
    cv: UploadFile = File(...),
    target_role: str = Form(...),
    location: str = Form(...),
):
    """
    Start SkillGap AI.

    Runs:
    - Profile Analyzer
    - Job Scout
    - Supervisor
    - Job Validation

    Then pauses at Human-in-the-Loop
    and returns jobs to the frontend.
    """

    file_bytes = await read_and_validate_pdf(
        cv
    )

    cv_text = extract_pdf_text(
        file_bytes
    )

    if not cv_text:
        raise HTTPException(
            status_code=400,
            detail=(
                "No readable text was found "
                "inside the PDF."
            ),
        )

    thread_id = str(
        uuid.uuid4()
    )

    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    initial_state = {
        "cv_text": cv_text,
        "target_role": target_role.strip(),
        "location": location.strip(),
        "execution_logs": [],
    }

    try:
        result = graph.invoke(
            initial_state,
            config=config,
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "SkillGap workflow failed: "
                f"{type(error).__name__}"
            ),
        ) from error

    if result.get("error_message"):
        raise HTTPException(
            status_code=400,
            detail=result["error_message"],
        )

    interrupt_payload = (
        get_interrupt_payload(result)
    )

    if interrupt_payload:
        return {
            "status":
                "waiting_for_job_selection",

            "thread_id":
                thread_id,

            "limited_results":
                interrupt_payload.get(
                    "limited_results",
                    False,
                ),

            "jobs":
                interrupt_payload.get(
                    "jobs",
                    [],
                ),

            "execution_logs":
                result.get(
                    "execution_logs",
                    [],
                ),
        }

    if result.get("final_report"):
        return {
            "status": "completed",
            "thread_id": thread_id,
            "result": result,
        }

    raise HTTPException(
        status_code=500,
        detail=(
            "Workflow did not return job selection "
            "or a final result."
        ),
    )


# ==========================================
# Resume After Job Selection
# ==========================================

@app.post("/api/select-job")
def select_job(
    request: JobSelectionRequest,
):
    """
    Resume the same LangGraph execution after
    the user selects a job.

    Runs the remaining workflow:
    Requirements → Gap Analyzer →
    Career Coach → Supervisor → Final Report
    """

    config = {
        "configurable": {
            "thread_id":
                request.thread_id,
        }
    }

    try:
        result = graph.invoke(
            Command(
                resume=request.selected_job_index
            ),
            config=config,
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Could not resume SkillGap workflow: "
                f"{type(error).__name__}"
            ),
        ) from error

    interrupt_payload = (
        get_interrupt_payload(result)
    )

    if interrupt_payload:
        return {
            "status":
                "waiting_for_job_selection",

            "thread_id":
                request.thread_id,

            "limited_results":
                interrupt_payload.get(
                    "limited_results",
                    False,
                ),

            "jobs":
                interrupt_payload.get(
                    "jobs",
                    [],
                ),

            "message":
                "Please select a valid job number.",

            "execution_logs":
                result.get(
                    "execution_logs",
                    [],
                ),
        }

    if result.get("error_message"):
        raise HTTPException(
            status_code=400,
            detail=result["error_message"],
        )

    return {
        "status": "completed",

        "thread_id":
            request.thread_id,

        "selected_job":
            result.get(
                "selected_job",
                {},
            ),

        "candidate_profile":
            result.get(
                "candidate_profile",
                {},
            ),

        "job_requirements":
            result.get(
                "job_requirements",
                {},
            ),

        "gap_analysis":
            result.get(
                "gap_analysis",
                {},
            ),

        "recommendations":
            result.get(
                "recommendations",
                {},
            ),

        "final_report":
            result.get(
                "final_report",
                "",
            ),

        "execution_logs":
            result.get(
                "execution_logs",
                [],
            ),
    }


# ==========================================
# Run
# ==========================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
