from crewai import Agent, Task, Crew, Process
from agent.config import get_llm

def run_insurance_crew(topic: str):
    """Runs a multi-agent collaboration to produce an insurance report."""
    llm = get_llm()

    # 1. Researcher Agent
    researcher = Agent(
        role='Insurance Fraud Researcher',
        goal='Analyze the latest trends and patterns in {topic}',
        backstory='You are a senior analyst with 15 years of experience in forensic auditing and insurance investigations.',
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

    # 2. Writer Agent
    writer = Agent(
        role='Technical Insurance Writer',
        goal='Write a professional and concise report about {topic} based on the research findings.',
        backstory='You specialize in making complex technical data accessible to insurance executives.',
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

    # Tasks
    research_task = Task(
        description=f"Conduct a deep dive into {topic}. Identify 3 key risk factors.",
        expected_output="A list of 3 detailed risk factors with examples.",
        agent=researcher
    )

    writing_task = Task(
        description=f"Format the research findings into a 1-page executive summary about {topic}.",
        expected_output="A Markdown formatted report.",
        agent=writer
    )

    # Crew
    crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, writing_task],
        process=Process.sequential
    )

    return crew.kickoff()
