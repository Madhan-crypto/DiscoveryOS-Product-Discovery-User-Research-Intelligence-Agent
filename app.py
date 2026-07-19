import os
from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

# NOTE: For deployment, use environment variables for keys
# In Streamlit/HuggingFace, add GEMINI_API_KEY to their 'Secrets' settings

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.1)

class DiscoveryState(TypedDict):
    raw_data: str
    extracted_insights: List[Dict]
    prioritized_roadmap: str

class InsightExtraction(BaseModel):
    pain_point: str = Field(description="The core problem or friction point identified.")
    category: str = Field(description="Categorization: 'UX/UI', 'Performance', 'Bug', 'Feature Request', or 'Billing'.")
    user_segment: str = Field(description="Type of user affected.")
    severity: str = Field(description="Impact level: High, Medium, Low.")
    confidence: float = Field(description="Confidence score.")

structured_llm = llm.with_structured_output(InsightExtraction)

def extraction_agent(state: DiscoveryState):
    prompt = f"Analyze this user feedback: {state['raw_data']}"
    response = structured_llm.invoke(prompt)
    return {"extracted_insights": [response.model_dump()]}

def prioritization_agent(state: DiscoveryState):
    insights_summary = "\n".join([f"- [{x['category']}] {x['pain_point']}" for x in state['extracted_insights']])
    prompt = f"Provide a Product Discovery Report for these insights: {insights_summary}"
    response = llm.invoke(prompt)
    return {"prioritized_roadmap": response.content}

def create_agent():
    workflow = StateGraph(DiscoveryState)
    workflow.add_node("extractor", extraction_agent)
    workflow.add_node("prioritizer", prioritization_agent)
    workflow.set_entry_point("extractor")
    workflow.add_edge("extractor", "prioritizer")
    workflow.add_edge("prioritizer", END)
    return workflow.compile()

if __name__ == "__main__":
    agent = create_agent()
    print("DiscoveryOS Agent Loaded Successfully.")
