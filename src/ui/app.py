import streamlit as st
import requests
import json

# --- CONFIGURATION ---
API_URL = "http://localhost:8000/chat"
st.set_page_config(page_title="Aether Store AI", page_icon="🤖", layout="wide")

# --- HEADER ---
st.title("🤖 Aether Store Assistant")
st.markdown("""
<style>
    .stChatMessage {
        background-color: #f0f2f6; 
        border-radius: 10px;
        padding: 10px;
    }
    .source-card {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 10px;
        margin-top: 5px;
        background-color: #ffffff;
        font-size: 0.9em;
    }
    .source-title { font-weight: bold; color: #0078D4; }
    .source-price { color: #28a745; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE (Memory) ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- SIDEBAR (Debug Info) ---
with st.sidebar:
    st.header("🔧 System Status")
    if st.button("Check API Health"):
        try:
            res = requests.get("http://localhost:8000/")
            if res.status_code == 200:
                data = res.json()
                st.success(f"Online: {data.get('mode', 'Unknown')}")
                st.info(f"Model: {data.get('chat_model', 'Unknown')}")
            else:
                st.error(f"Error: {res.status_code}")
        except Exception as e:
            st.error(f"Connection Failed: {e}")
    
    st.markdown("---")
    st.markdown("**Architecture:**")
    st.markdown("- **UI:** Streamlit (Python)")
    st.markdown("- **API:** FastAPI (Docker)")
    st.markdown("- **DB:** SQL Server 2025")
    st.markdown("- **LLM:** Hybrid (Ollama/OpenAI)")

# --- CHAT INTERFACE ---
# 1. Display History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("🔍 View Sources Used"):
                for src in message["sources"]:
                    st.markdown(f"""
                    <div class="source-card">
                        <div class="source-title">{src['title']}</div>
                        <div class="source-price">${src['price']}</div>
                        <div>{src['distance']:.4f} similarity</div>
                    </div>
                    """, unsafe_allow_html=True)

# 2. Input Logic
if prompt := st.chat_input("Ask about our products..."):
    # Add User Message to History
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call API
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        try:
            with st.spinner("Thinking..."):
                response = requests.post(API_URL, json={"question": prompt})
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "No answer provided.")
                    sources = data.get("sources", [])
                    
                    message_placeholder.markdown(answer)
                    
                    # Display Sources nicely
                    if sources:
                        with st.expander("🔍 View Sources Used"):
                            for src in sources:
                                st.markdown(f"""
                                <div class="source-card">
                                    <div class="source-title">{src['title']}</div>
                                    <div class="source-price">${src['price']}</div>
                                    <div style="color: #666; font-size: 0.8em;">Distance: {src['distance']:.4f}</div>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    # Save Assistant Message to History
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "sources": sources
                    })
                else:
                    message_placeholder.error(f"API Error: {response.status_code}")
        except Exception as e:
            message_placeholder.error(f"Connection Error: {e}")