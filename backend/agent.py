"""
The agent itself: takes a user message, pulls relevant long-term memories,
builds a prompt with that context, calls the LLM, and saves the new
exchange back into memory.
"""

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from .memory import MemoryStore

SYSTEM_PROMPT = """You are a helpful personal assistant with long-term memory.
You remember past conversations with this user and use that context to give
more personal, consistent, and useful answers. If relevant memories are
provided below, use them naturally — don't just repeat them verbatim.
If nothing relevant is remembered, just answer normally.
"""


class MemoryAgent:
    def __init__(self, user_id: str, openai_api_key: str = None, model: str = "gemini-2.5-flash"):
        # openai_api_key is optional — only used if you want OpenAI embeddings
        # for memory search instead of the free local embedding model.
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.memory = MemoryStore(user_id=user_id, openai_api_key=self.openai_api_key)

        google_api_key = os.getenv("GOOGLE_API_KEY")
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=google_api_key,
            temperature=0.7,
        )
        # Short-term (this-session) conversation history, separate from
        # long-term retrieved memory.
        self.session_history: list = []

    def chat(self, user_message: str, k_memories: int = 5) -> str:
        # 1. Pull relevant long-term memories for this query
        relevant = self.memory.retrieve_relevant(user_message, k=k_memories)
        memory_context = "\n\n".join(relevant) if relevant else "No relevant past memories."

        # 2. Build the message list: system prompt + retrieved memory + session history + new message
        messages = [
            SystemMessage(content=f"{SYSTEM_PROMPT}\n\nRelevant memories:\n{memory_context}")
        ]
        messages.extend(self.session_history[-10:])  # last 10 turns of this session
        messages.append(HumanMessage(content=user_message))

        # 3. Call the LLM
        response = self.llm.invoke(messages)
        reply = response.content

        # 4. Update session history
        self.session_history.append(HumanMessage(content=user_message))
        self.session_history.append(AIMessage(content=reply))

        # 5. Persist this exchange to long-term memory
        self.memory.add_memory(user_message, reply)

        return reply
