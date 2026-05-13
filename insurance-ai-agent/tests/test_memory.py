from agent.memory import add_to_memory, query_memory
import time

def test_memory_persistence():
    unique_text = f"Test Memory Entry {time.time()}"
    add_to_memory(unique_text, {"source": "pytest"})
    
    # Retrieve
    context = query_memory("Test Memory Entry")
    assert unique_text in context
