import os
from typing import List, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

# قراءة المتغيرات من ملف .env
load_dotenv()


# ==========================================
# 1. Shared State Definition
# ==========================================
class AgentState(TypedDict):
    # User Inputs
    cv_text: str
    target_role: str
    target_location: str
    
    # Profile Analyzer Outputs
    candidate_skills: List[str]
    candidate_experience_years: int
    candidate_summary: str

    # Job Scout Agent Outputs
    job_description: str
    job_requirements: List[str]
    job_title_found: str
    
    # Supervisor & Workflow Control Flags
    review_stage: Optional[str]
    supervisor_decision: Optional[str]
    supervisor_feedback: Optional[str]
    profile_retry_count: int


# ==========================================
# 2. Structured Output Schema
# ==========================================
class JobScoutOutput(BaseModel):
    job_title: str = Field(
        description="The formal title of the target job position."
    )
    job_description: str = Field(
        description="A clear summary of the job responsibilities and scope."
    )
    key_requirements: List[str] = Field(
        description="List of essential skills, tools, and qualifications required for this role."
    )


# ==========================================
# 3. Job Scout Agent Node
# ==========================================
def job_scout_agent(state: AgentState) -> dict:
    # جلب المفتاح المتاح
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError("لم يتم العثور على API Key! تأكدي من ضبط المتغيرات البيئية.")

    # Initialize LLM via OpenRouter
    llm = ChatOpenAI(
        model="openai/gpt-4o-mini",
        temperature=0.2,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )
    structured_llm = llm.with_structured_output(JobScoutOutput)

    # Fetch inputs from state
    target_role = state.get("target_role", "Software Engineer")
    target_location = state.get("target_location", "Riyadh")

    # System Prompt
    system_prompt = (
        "You are an expert Talent Acquisition Specialist and Job Market Scout.\n"
        "Your role is to analyze a target job title and location, then generate a realistic, standard market profile "
        "including job title, core description, and essential technical/soft skill requirements."
    )

    human_prompt = f"Target Role: {target_role}\nTarget Location: {target_location}"

    # Invoke LLM
    result: JobScoutOutput = structured_llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ])

    return {
        "job_title_found": result.job_title,
        "job_description": result.job_description,
        "job_requirements": result.key_requirements,
        "review_stage": "job_scout"
    }


# ==========================================
# 4. Local Execution Test
# ==========================================
if __name__ == "__main__":
    test_state: AgentState = {
        "cv_text": "",
        "target_role": "AI Engineer",
        "target_location": "Riyadh",
        "candidate_skills": [],
        "candidate_experience_years": 0,
        "candidate_summary": "",
        "job_description": "",
        "job_requirements": [],
        "job_title_found": "",
        "review_stage": None,
        "supervisor_decision": None,
        "supervisor_feedback": None,
        "profile_retry_count": 0
    }

    print("--- Running Job Scout Agent ---")
    output = job_scout_agent(test_state)
    
    print("\nJob Title:", output["job_title_found"])
    print("\nJob Description:\n", output["job_description"])
    print("\nKey Requirements:\n", output["job_requirements"])
