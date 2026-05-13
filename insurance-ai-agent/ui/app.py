import streamlit as st
import requests
import json

st.set_page_config(page_title="Insurance AI Assistant", page_icon="🤖", layout="wide")

st.title("🛡️ Insurance AI Agent Dashboard")
st.sidebar.title("Configuration")

api_url = st.sidebar.text_input("FastAPI URL", "http://localhost:8000")

tab1, tab2 = st.tabs(["💬 Agent Chat (ReAct)", "👥 Multi-Agent (CrewAI)"])

with tab1:
    st.header("LangChain ReAct Agent")
    st.write("Reasoning -> Action -> Observation -> Final Answer")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about an insurance claim or fraud risk..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Agent is thinking..."):
                try:
                    response = requests.post(f"{api_url}/chat", json={"text": prompt})
                    if response.status_code == 200:
                        ans = response.json()["response"]
                        st.markdown(ans)
                        st.session_state.messages.append({"role": "assistant", "content": ans})
                    else:
                        st.error(f"Error: {response.text}")
                except Exception as e:
                    st.error(f"Connection error: {e}")

with tab2:
    st.header("CrewAI Multi-Agent Workflow")
    topic = st.text_input("Report Topic", "Automobile Insurance Fraud Trends 2024")
    if st.button("Start Crew Workflow"):
        with st.spinner("Crew is working (Researcher -> Writer)..."):
            try:
                res = requests.post(f"{api_url}/crew/run", json={"topic": topic})
                if res.status_code == 200:
                    st.markdown("### Final Report")
                    st.markdown(res.json()["result"])
                else:
                    st.error(f"Error: {res.text}")
            except Exception as e:
                st.error(f"Connection error: {e}")

st.sidebar.markdown("---")
if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    st.rerun()
