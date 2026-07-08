import streamlit as st
import sys
import os
import re

# 1. THEME & LAYOUT
st.set_page_config(layout="wide", page_title="Project Drona", page_icon="🏏")

# Custom CSS for dark theme
st.markdown("""
<style>
.stApp {
    background-color: #0e1117;
    color: #c9d1d9;
}
</style>
""", unsafe_allow_html=True)

# Ensure src is in the path to import agent modules
sys.path.append(os.path.abspath("src"))
import agent as drona_agent
from langchain_ollama import OllamaLLM

# Setup session state for history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "match_context" not in st.session_state:
    st.session_state.match_context = ""

# Monkey Patch Tools for Observation streaming inside st.status
if not getattr(drona_agent, "_patched_tools", False):
    original_tool_map = dict(drona_agent.TOOL_MAP)
    for tool_name, tool_fn in original_tool_map.items():
        def make_wrapper(name, fn):
            def wrapper(input_str):
                obs = fn(input_str)
                # This will print inside the st.status block where the tool is called
                st.markdown(f"**Observation ({name}):**\n```json\n{str(obs)[:800]}...\n```")
                return obs
            return wrapper
        drona_agent.TOOL_MAP[tool_name] = make_wrapper(tool_name, tool_fn)
    drona_agent._patched_tools = True

# Monkey patch OllamaLLM to stream Thoughts/SQL from the ReAct loop
if not getattr(drona_agent, "_patched_llm", False):
    original_invoke = OllamaLLM.invoke
    def streaming_invoke(self, prompt, *args, **kwargs):
        res = original_invoke(self, prompt, *args, **kwargs)
        # We only want to stream if it's the ReAct agent, not the Coach narration
        if isinstance(prompt, str) and "STRICT FORMAT" in prompt:
            if isinstance(res, str):
                # Filter out Final Answer from the status expander
                display_text = re.sub(r"(?i)Final Answer:.*", "", res, flags=re.DOTALL).strip()
                if display_text:
                    st.markdown(f"```text\n{display_text}\n```")
        return res
    OllamaLLM.invoke = streaming_invoke
    drona_agent._patched_llm = True

if "llm" not in st.session_state:
    llm = OllamaLLM(
        model="drona-v2", # Default to fine-tuned model
        base_url="http://localhost:11434",
        temperature=0.1, 
        num_predict=600,
    )
    st.session_state.llm = llm
    st.session_state.agent = llm

# 2. MATCH CONTEXT SIDEBAR
with st.sidebar:
    st.header("🏟️ Match State")
    venue = st.selectbox("Venue", [
        "Wankhede Stadium", "Eden Gardens", "MA Chidambaram Stadium", 
        "Feroz Shah Kotla", "M Chinnaswamy Stadium", "DY Patil Stadium", 
        "Rajiv Gandhi Intl Stadium", "Sawai Mansingh Stadium", 
        "Punjab Cricket Association Stadium", "Brabourne Stadium"
    ])
    phase = st.selectbox("Match Phase", ["powerplay", "middle", "death"])
    situation = st.text_area("Situation Details", placeholder="Target 185, defending 34 off 18")
    
    if st.button("Update Context", use_container_width=True):
        st.session_state.match_context = f"Venue: {venue} | Phase: {phase} | Situation: {situation}"
        st.success("Context updated silently.")

# 5. Display CHAT HISTORY
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 4. QUICK ACTION PROMPTS
st.write("")
col1, col2, col3 = st.columns(3)
quick_action = None
if col1.button("Best death bowlers here?", use_container_width=True):
    quick_action = "Who are the best death bowlers at this venue?"
if col2.button("Who counters V Kohli?", use_container_width=True):
    quick_action = "Who counters V Kohli effectively?"
if col3.button("Slowest surface behavior?", use_container_width=True):
    quick_action = "Which venue has the slowest surface behavior index?"

# User Input
user_query = st.chat_input("Ask Drona for tactical advice...")

# Determine active prompt
active_prompt = quick_action if quick_action else user_query

if active_prompt:
    # Build full prompt
    full_prompt = active_prompt
    if st.session_state.match_context:
        full_prompt = f"CONTEXT: {st.session_state.match_context}\n\nQUESTION: {active_prompt}"
        
    st.session_state.messages.append({"role": "user", "content": active_prompt})
    with st.chat_message("user"):
        st.markdown(active_prompt)

    with st.chat_message("assistant"):
        # 3. AGENT TRANSPARENCY (THE "BRAIN" EXPANDER)
        with st.status("🧠 Drona is analyzing matchups...", expanded=True) as status:
            lessons = drona_agent.load_lessons(venue=st.session_state.match_context, phase="")
            
            # Use invoke_agent inside the status block to capture the monkey-patched output
            agent_out = drona_agent.invoke_agent(st.session_state.agent, full_prompt, lessons)
            
            sql_result = agent_out.get("sql_result", {})
            if agent_out.get("error") and not sql_result:
                coach_advice = f"Agent error: {agent_out['error']}"
            else:
                coach_advice = drona_agent.narrate_as_coach(
                    st.session_state.llm, 
                    st.session_state.match_context or full_prompt, 
                    sql_result, 
                    lessons
                )
            
            status.update(label="Analysis Complete", state="complete", expanded=False)
        
        # Display Final Answer cleanly
        final_answer = agent_out.get("output", "")
        
        output_md = f"**Final Answer:**\n{final_answer}\n\n"
        if coach_advice and "Agent error" not in coach_advice:
            output_md += f"**Coach Drona says:**\n> {coach_advice}"
            
        st.markdown(output_md)
        st.session_state.messages.append({"role": "assistant", "content": output_md})
