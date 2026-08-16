# SkillGap AI

**SkillGap AI** is a multi-agent career analysis system that compares a candidate's CV with real job opportunities, identifies verified skill gaps, and generates a practical development plan.

The system uses **LangGraph** to orchestrate multiple specialized agents, **OpenRouter** for LLM access, **Tavily** for live job discovery, and **FastAPI** to expose the workflow through a lightweight web interface.

---

## Overview

SkillGap AI helps a user answer three practical questions:

1. **Which current jobs are relevant to my target role and location?**
2. **Which required skills do I already have, and which ones are missing?**
3. **What should I learn or build next to improve my chances?**

The user uploads a PDF CV, enters a target role and location, reviews real job opportunities found by the system, selects one job, and receives a grounded skill-gap analysis and career development plan.

---

## Key Features

- PDF CV upload and text extraction
- Multi-layer PDF validation
- Structured candidate profile analysis
- Live job search using Tavily
- Parallel execution of profile analysis and job discovery
- Job-result validation and search refinement
- Human-in-the-Loop job selection
- Job requirement extraction
- Deterministic skill-gap comparison
- Skill coverage calculation
- Evidence-gap detection for skills listed without clear supporting experience
- Prioritized learning recommendations
- Portfolio project recommendation
- Supervisor-based validation and retry logic
- Controlled failure handling
- LangGraph checkpointing for interrupted/resumed workflows
- Input validation and bounded user inputs
- FastAPI backend with a lightweight web frontend
- Automated tests for validation and workflow routing

---

## Live Demo

The application is deployed on Render and can be accessed here:

[SkillGap AI - Live Demo](https://skillgap-ai-a336.onrender.com)
---

## Multi-Agent Architecture

The system is composed of specialized agents, each responsible for a specific stage of the workflow.

| Component | Responsibility |
|---|---|
| **Profile Analyzer** | Extracts skills, projects, experience, education, skill evidence, and estimated experience level from the CV. |
| **Job Scout** | Searches for current job opportunities matching the target role and location using Tavily. |
| **Requirements Agent** | Extracts required skills, preferred skills, tools, frameworks, and other structured requirements from the selected job. |
| **Gap Analyzer** | Deterministically compares verified candidate skills with job requirements and calculates skill coverage. |
| **Career Coach** | Converts verified gaps into prioritized learning steps, a portfolio project, and an application recommendation. |
| **Supervisor** | Reviews selected agent outputs and decides whether to approve, retry, or stop with a controlled failure. |

---

## Workflow

```mermaid
flowchart TD
    A[Upload CV + Target Role + Location] --> B[Input Guard]
    B --> C[Parallel Start]

    C --> D[Profile Analyzer]
    C --> E[Job Scout]

    D --> F[Supervisor Review]
    F -->|Approved| G[Profile Ready]
    F -->|Retry| D

    E --> H[Job Validation]
    H -->|Enough Results| I[Jobs Ready]
    H -->|Limited Results| I
    H -->|Retry Search| E

    G --> J[Human-in-the-Loop Job Selection]
    I --> J

    J --> K[Selected Job Enrichment]
    K --> L[Requirements Agent]
    L --> M[Supervisor Review]
    M -->|Approved| N[Gap Analyzer]
    M -->|Retry| L

    N --> O[Career Coach]
    O --> P[Supervisor Review]
    P -->|Approved| Q[Final Report]
    P -->|Retry| O

    B -->|Invalid Input| X[Controlled Failure]
    H -->|Failure| X
    F -->|Failure| X
    M -->|Failure| X
    P -->|Failure| X
```

### Parallel Execution

After input validation, the workflow fans out into two independent branches:

- **Profile Analyzer** analyzes the CV.
- **Job Scout** searches for relevant job opportunities.

LangGraph waits for both branches to become ready before continuing to the Human-in-the-Loop job selection step.

---

## Human-in-the-Loop

SkillGap AI does not automatically choose a job for the user.

After the first stage completes, the LangGraph execution is paused using `interrupt()`. The frontend displays the available jobs, and the user chooses the opportunity they want to analyze.

The same workflow is then resumed using the stored `thread_id`, preserving the graph state through LangGraph checkpointing.

---

## Skill Gap Analysis

The Gap Analyzer uses deterministic comparison rather than asking an LLM to invent a match score.

It:

- normalizes skill names and common aliases
- removes duplicate skills
- identifies matching required skills
- identifies missing required skills
- separates preferred-skill matches
- detects skills that appear in the CV but have no clear supporting evidence
- calculates required-skill coverage

This keeps the final recommendations grounded in data extracted from the CV and the selected job.

---

## Tech Stack

### Backend and Orchestration

- Python
- FastAPI
- LangGraph
- LangChain
- Pydantic
- Uvicorn

### AI and Search

- OpenRouter
- OpenAI-compatible `ChatOpenAI`
- Tavily Search API

### File Processing

- PyPDF

### Frontend

- HTML
- CSS
- JavaScript

### Testing

- Pytest
- Pytest Asyncio

---

## Project Structure

```text
SkillGap-AI/
├── agents/
│   ├── __init__.py
│   ├── career_coach.py
│   ├── gap_analyzer.py
│   ├── job_scout.py
│   ├── profile_analyzer.py
│   ├── requirements_agent.py
│   └── supervisor.py
│
├── workflow/
│   ├── __init__.py
│   ├── analysis_nodes.py
│   ├── graph.py
│   ├── input_nodes.py
│   ├── job_nodes.py
│   ├── output_nodes.py
│   ├── runtime.py
│   ├── state.py
│   └── supervision_nodes.py
│
├── frontend/
│   ├── index.html
│   ├── styles.css
│   ├── renderers.js
│   └── app.js
│
├── tests/
│   ├── test_api_helpers.py
│   ├── test_input_guard.py
│   ├── test_job_validation.py
│   └── test_routing.py
│
├── .env.example
├── .gitignore
├── api.py
├── api_helpers.py
├── cli.py
├── requirements.txt
├── skillgap_ai.py
└── README.md
```

### Code Organization

The project separates responsibilities across several layers:

- `agents/` contains the specialized AI and deterministic analysis components.
- `workflow/` contains LangGraph state, nodes, routing, supervision, and graph construction.
- `frontend/` contains the web interface, styles, rendering logic, and client-side application logic.
- `tests/` contains automated tests for input validation, PDF handling, job validation, and workflow routing.
- `api.py` exposes the application through FastAPI.
- `api_helpers.py` handles PDF validation, text extraction, and API helper logic.
- `skillgap_ai.py` provides the main workflow entry point.
- `cli.py` provides command-line execution of the workflow.

---

## Environment Variables

Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_MODEL=your_openrouter_model
TAVILY_API_KEY=your_tavily_api_key
```

You can copy the provided example file.

### macOS / Linux

```bash
cp .env.example .env
```

### Windows PowerShell

```powershell
Copy-Item .env.example .env
```

Then add your API keys and preferred OpenRouter model.

> `.env` is excluded from version control. Never commit API keys or other secrets to the repository.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/mnarzaid247-stack/SkillGap-AI.git
cd SkillGap-AI
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**Windows PowerShell**

```powershell
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure environment variables

Create `.env` from `.env.example` and configure:

- `OPENROUTER_API_KEY`
- `OPENROUTER_MODEL`
- `TAVILY_API_KEY`

---

## Running the Application

Start the FastAPI server from the project root:

```bash
uvicorn api:app --reload
```

Then open:

```text
http://127.0.0.1:8000
```

FastAPI interactive documentation is available at:

```text
http://127.0.0.1:8000/docs
```

The health endpoint is available at:

```text
http://127.0.0.1:8000/api/health
```

---

## Running the Tests

Run the full automated test suite from the project root:

```bash
python -m pytest -v
```

The test suite covers:

- input validation
- CV length validation
- target role and location validation
- PDF extension and content-type validation
- PDF size and parsing validation
- job URL validation
- job relevance filtering
- closed-job detection
- duplicate job filtering
- workflow routing
- retry-limit routing
- Supervisor routing

The current test suite contains **52 passing tests**.

---

## API Endpoints

### Health Check

```http
GET /api/health
```

Returns the current API status.

### Start Analysis

```http
POST /api/start
```

Form fields:

- `cv` — PDF CV file
- `target_role` — desired role
- `location` — desired job location

The endpoint validates the uploaded CV, extracts its text, starts the LangGraph workflow, and normally returns:

- a `thread_id`
- discovered jobs
- workflow execution logs
- a status indicating that the graph is waiting for job selection

### Select Job and Resume Workflow

```http
POST /api/select-job
```

JSON body:

```json
{
  "thread_id": "your-thread-id",
  "selected_job_index": 1
}
```

This resumes the interrupted LangGraph workflow and runs the remaining analysis.

---

## Input and File Validation

The application validates user input before expensive AI or search operations are executed.

### CV Validation

Uploaded CVs are checked for:

- `.pdf` file extension
- recognized PDF content type
- non-empty file content
- maximum file size of **10 MB**
- successful parsing by PyPDF
- extractable text content

The extracted CV text is also bounded before entering the workflow.

### User Input Validation

The Input Guard validates:

- CV text presence and length
- target role presence and maximum length
- location presence and maximum length

Invalid input is routed to a controlled failure path instead of continuing through the multi-agent workflow.

---

## Frontend Safety

Dynamic data returned by job search and AI components is rendered using DOM APIs such as `textContent` and `createTextNode` rather than injecting external content through `innerHTML`.

This reduces the risk of rendering untrusted job or model output as executable HTML.

---

## Final Output

The completed analysis includes information such as:

- candidate profile summary
- detected skills
- estimated experience level
- selected job and company
- required and preferred skills
- matching skills
- missing required skills
- evidence gaps
- skill coverage percentage
- priority skill gaps
- recommended learning order
- recommended portfolio project
- next action
- application recommendation

---

## Validation and Reliability

SkillGap AI includes several safeguards to reduce unsupported or unreliable outputs:

- structured Pydantic outputs for LLM agents
- deterministic validation of important fields
- explicit skill evidence extraction
- deterministic skill matching and coverage calculation
- job-result validation
- bounded user input
- multi-layer PDF validation
- limited retry counters
- Supervisor review stages
- controlled failure paths when outputs cannot be validated
- Human-in-the-Loop job selection
- automated tests for critical validation and routing logic
- no automatic invention of job location when the search result does not provide one reliably

---

## Current Limitations

- The current version accepts **PDF CV files only**.
- Scanned PDFs without extractable text may not work because the application does not currently perform OCR.
- Job quality and availability depend on search results returned by Tavily.
- LLM output quality depends on the OpenRouter model configured in `.env`.
- The current checkpoint store uses LangGraph's in-memory saver, so workflow state is not intended as persistent production storage.
- The current application does not provide user accounts or persistent analysis history.

---

## Requirements

The project uses pinned dependency versions in `requirements.txt` to keep local and deployment environments reproducible.

Main dependencies include:

```text
langchain-core
langchain-openai
pydantic
typing-extensions
python-dotenv
tavily-python
langgraph
fastapi
uvicorn
python-multipart
pypdf
pytest
pytest-asyncio
```

---

## Future Improvements

Possible next steps include:

- persistent checkpoint storage
- authentication and user accounts
- saved analysis history
- support for DOCX CVs
- OCR support for scanned CVs
- improved job-source filtering
- persistent result storage
- production monitoring and observability


---

## SDAIAAcademy details

This project was developed as part of the **Advanced Agentic AI Systems Engineering** training program by **SDAIA Academy**.

The project applies concepts covered during the program, including multi-agent systems, agent orchestration, Human-in-the-Loop workflows, structured LLM outputs, validation, and supervisor-based agent coordination.

### SDAIA Academy

GitHub: https://github.com/SDAIAAcademy

