from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agent.agent import get_insurance_agent
from agent.memory import add_to_memory, query_memory
from multi_agents.crew import run_insurance_crew
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Insurance AI Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    text: str

class CrewRequest(BaseModel):
    topic: str

@app.get("/health")
def health():
    return {"status": "ready"}

@app.post("/chat")
async def chat(message: ChatMessage):
    try:
        agent = get_insurance_agent()
        # Search memory for context
        context = query_memory(message.text)
        
        full_prompt = f"Previous Context: {context}\nUser Message: {message.text}"
        
        response = agent.invoke({"input": full_prompt})
        output = response["output"]
        
        # Save to memory
        add_to_memory(f"User: {message.text} | AI: {output}")
        
        return {"response": output}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/crew/run")
async def run_crew(request: CrewRequest):
    try:
        result = run_insurance_crew(request.topic)
        return {"result": str(result)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
