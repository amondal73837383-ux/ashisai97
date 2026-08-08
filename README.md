# Custom AI Agent with Memory

Voice/text assistant with long-term memory, built with LangChain, OpenAI, Whisper, ChromaDB, and Streamlit.

## How it works

1. **Input** — type a message, or record your voice (transcribed via Whisper).
2. **Memory retrieval** — before answering, the agent searches ChromaDB for
   semantically relevant past exchanges (not just the last few messages —
   *any* past conversation that's related to your current question).
3. **Response generation** — the retrieved memories + recent session history
   + your new message go to the LLM (GPT-4o-mini by default) via LangChain.
4. **Memory write-back** — the new exchange is embedded and stored in
   ChromaDB so future conversations can recall it.

This is what makes it "long-term" memory rather than just a chat window with
scrollback: the agent can recall something from 50 conversations ago if it's
relevant, even though it's long since scrolled out of the visible context window.

## Project structure

```
memory-agent/
├── backend/
│   ├── memory.py      # ChromaDB-backed memory store (add/retrieve/clear)
│   ├── agent.py        # Core agent: retrieval + LLM call + memory write-back
│   └── transcribe.py    # Whisper speech-to-text wrapper
├── frontend/
│   └── app.py          # Streamlit chat UI (text + voice)
├── requirements.txt
└── .env.example
```

## Setup

```bash
cd memory-agent
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and add your OPENAI_API_KEY
```

## Run

```bash
streamlit run frontend/app.py
```

Open the local URL Streamlit prints (usually http://localhost:8501).

## Using it without the UI (just Python)

```python
from backend.agent import MemoryAgent

agent = MemoryAgent(user_id="ashish", openai_api_key="sk-...")

print(agent.chat("My favorite programming language is Rust."))
print(agent.chat("What did I tell you about my favorite language?"))
# -> It should recall "Rust" even in a brand new session, because
#    that exchange was persisted to ChromaDB, not just kept in RAM.
```

## Customization ideas

- **Swap ChromaDB for Redis**: use `redis` + `redisearch`/`RedisVL` for
  vector search if you need faster reads at scale or already run Redis.
- **Swap Streamlit for Next.js**: expose `MemoryAgent.chat()` behind a
  FastAPI endpoint and call it from a Next.js frontend for a more
  polished UI.
- **Multi-user**: the `user_id` param already isolates memory per user —
  point it at your auth system's user ID.
- **Memory decay/summarization**: for very long-lived agents, periodically
  summarize old memories into condensed "profile facts" so retrieval stays
  fast and relevant instead of growing unbounded.
- **Different LLM**: change the `model` param in `MemoryAgent` (e.g. to
  `gpt-4o` for higher quality, or a local model via Ollama + LangChain's
  `ChatOllama` for a fully offline agent).

## Notes on costs

- Whisper transcription and OpenAI embeddings/chat completions are billed
  per API call — check current OpenAI pricing before heavy testing.
- ChromaDB runs locally and is free; it persists to disk in `chroma_db/`.
