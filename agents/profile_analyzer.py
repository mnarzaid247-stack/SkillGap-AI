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
    
    # Profile Analyzer Agent Outputs
    candidate_skills: List[str]
    candidate_experience_years: int
    candidate_summary: str
    
    # Supervisor & Workflow Control Flags
    review_stage: Optional[str]
    supervisor_decision: Optional[str]
    supervisor_feedback: Optional[str]
    profile_retry_count: int


# ==========================================
# 2. Structured Output Schema
# ==========================================
class ProfileAnalysisOutput(BaseModel):
    skills: List[str] = Field(
        description="List of all technical and soft skills extracted from the CV."
    )
    experience_years: int = Field(
        description="Total years of work or academic experience derived from the CV."
    )
    summary: str = Field(
        description="A concise professional summary highlighting the candidate's core strengths."
    )


# ==========================================
# 3. Profile Analyzer Agent Node
# ==========================================
def profile_analyzer_agent(state: AgentState) -> dict:
    # جلب المفتاح سواء كان باسم OPENROUTER_API_KEY أو OPENAI_API_KEY
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError("لم يتم العثور على API Key! تأكدي من وجود ملف .env وفيه المفتاح.")

    # Initialize LLM via OpenRouter
    llm = ChatOpenAI(
        model="openai/gpt-4o-mini",
        temperature=0,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )
    structured_llm = llm.with_structured_output(ProfileAnalysisOutput)

    # Fetch values from shared state
    cv_text = state.get("cv_text", "")
    target_role = state.get("target_role", "")
    feedback = state.get("supervisor_feedback", "")
    retry_count = state.get("profile_retry_count", 0)

    # System instructions
    system_prompt = (
        "You are an expert HR and technical resume analyzer.\n"
        "Your job is to extract skills, total years of experience, and a professional summary from the provided CV.\n"
        "Be accurate and do not fabricate any missing information."
    )

    if feedback and retry_count > 0:
        system_prompt += f"\n\nSupervisor correction feedback: {feedback}"

    human_prompt = f"Resume Text:\n{cv_text}\n\nTarget Role: {target_role}"

    # Invoke LLM
    result: ProfileAnalysisOutput = structured_llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ])

    return {
        "candidate_skills": result.skills,
        "candidate_experience_years": result.experience_years,
        "candidate_summary": result.summary,
        "review_stage": "profile_analyzer",
        "profile_retry_count": retry_count + 1 if feedback else retry_count
    }


# ==========================================
# 4. Local Execution Test
# ==========================================
if __name__ == "__main__":
    test_state: AgentState = {
        "cv_text": "Graduated in Computer Science. Skilled in Python, Linux, Shell Scripting, SQL, Oracle DB, and Agentic AI.",
        "target_role": "AI / DevOps Engineer",
        "target_location": "Riyadh",
        "candidate_skills": [],
        "candidate_experience_years": 0,
        "candidate_summary": "",
        "review_stage": None,
        "supervisor_decision": None,
        "supervisor_feedback": None,
        "profile_retry_count": 0
    }

    print("--- Running Profile Analyzer Agent ---")
    output = profile_analyzer_agent(test_state)
    
    print("\nExtracted Skills:", output["candidate_skills"])
    print("Years of Experience:", output["candidate_experience_years"])
    print("Summary:", output["candidate_summary"])
