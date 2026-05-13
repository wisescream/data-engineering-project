from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from .config import get_llm
from .tools import web_search, get_claim_history, predict_fraud_j3, compliance_check_j4, get_system_metrics

def get_insurance_agent():
    """L'Agent SANLAM intégré (J1-J5)."""
    llm = get_llm()
    tools = [web_search, get_claim_history, predict_fraud_j3, compliance_check_j4, get_system_metrics]
    
    # Custom instructions for the agent to use the integrated pipeline
    base_prompt = hub.pull("hwchase17/react")
    
    agent = create_react_agent(llm, tools, base_prompt)
    
    return AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        handle_parsing_errors=True,
        max_iterations=10
    )
