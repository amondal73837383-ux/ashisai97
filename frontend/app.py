"""
Streamlit UI for the Custom AI Agent with Memory.
Supports typed text and voice input (via mic recorder + Whisper).

Run with:  streamlit run frontend/app.py
"""

import os
import sys
import streamlit as st
from dotenv import load_dotenv

# Allow importing backend/ when running from project root
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from backend.agent import MemoryAgent
from backend.transcribe import transcribe_audio

load_dotenv()

st.set_page_config(page_title="Memory Agent", page_icon="🧠", layout="centered")
st.title("🧠 AI Agent with Long-Term Memory")
st.caption("Chat by text or voice — it remembers your past conversations.")

# --- Setup: API key (optional — only needed for voice transcription via Whisper) ---
api_key = os.getenv("OPENAI_API_KEY") or st.sidebar.text_input(
    "OpenAI API Key (optional, only for voice input)", type="password"
)
st.sidebar.caption("Chat runs free locally via Ollama. Add an OpenAI key only if you want voice input.")

user_id = st.sidebar.text_input("User ID", value="default_user",
                                 help="Separate memory is kept per user ID.")

if st.sidebar.button("🗑️ Clear all memory"):
    MemoryAgent(user_id=user_id, openai_api_key=api_key).memory.clear()
    st.sidebar.success("Memory cleared.")

# --- Init agent + chat state (once per session/user) ---
if "agent" not in st.session_state or st.session_state.get("user_id") != user_id:
    st.session_state.agent = MemoryAgent(user_id=user_id, openai_api_key=api_key)
    st.session_state.user_id = user_id
    st.session_state.messages = []

# --- Render chat history ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Voice input (requires an OpenAI key for Whisper transcription) ---
audio = None
user_input = None

if api_key:
    try:
        from streamlit_mic_recorder import mic_recorder
        audio = mic_recorder(start_prompt="🎤 Record", stop_prompt="⏹️ Stop", key="recorder")
    except ImportError:
        st.sidebar.info("Install streamlit-mic-recorder for voice input.")
else:
    st.sidebar.caption("🎤 Voice input disabled — add an OpenAI key above to enable it.")

if audio and audio.get("bytes"):
    with st.spinner("Transcribing..."):
        user_input = transcribe_audio(audio["bytes"], filename="audio.wav")
    st.chat_message("user").markdown(f"🎤 {user_input}")

# --- Text input ---
typed = st.chat_input("Type a message...")
if typed:
    user_input = typed
    st.chat_message("user").markdown(user_input)

# --- Process message ---
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = st.session_state.agent.chat(user_input)
        st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})
