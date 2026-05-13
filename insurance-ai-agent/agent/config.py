from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

def get_llm():
    """Returns a LangChain compatible LLM client for OpenRouter."""
    return ChatOpenAI(
        openai_api_base=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        model_name=os.getenv("MODEL_NAME", "meta-llama/llama-3-8b-instruct:free"),
        default_headers={
            "HTTP-Referer": "https://localhost:3000",
            "X-Title": "Insurance Agent TP"
        }
    )
