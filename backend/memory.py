"""
Long-term memory layer backed by ChromaDB.

Stores each exchange (user message + assistant reply) as a document with
an embedding, so we can later retrieve semantically relevant past
conversations to give the agent "long-term memory" beyond its context
window.
"""

import os
import uuid
from datetime import datetime

import chromadb
from chromadb.utils import embedding_functions


class MemoryStore:
    def __init__(self, user_id: str, persist_dir: str = None, openai_api_key: str = None):
        """
        user_id: lets you keep separate memory per user (multi-tenant).
        persist_dir: where Chroma writes its local database to disk.
        """
        persist_dir = persist_dir or os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        self.client = chromadb.PersistentClient(path=persist_dir)

        # Use OpenAI embeddings for semantic search quality.
        # Falls back to Chroma's default local embedding function if no key given.
        if openai_api_key:
            self.embed_fn = embedding_functions.OpenAIEmbeddingFunction(
                api_key=openai_api_key,
                model_name="text-embedding-3-small",
            )
        else:
            self.embed_fn = embedding_functions.DefaultEmbeddingFunction()

        self.collection = self.client.get_or_create_collection(
            name=f"memory_{user_id}",
            embedding_function=self.embed_fn,
        )

    def add_memory(self, user_message: str, assistant_message: str, metadata: dict = None):
        """Store one exchange as a retrievable memory."""
        doc_id = str(uuid.uuid4())
        text = f"User: {user_message}\nAssistant: {assistant_message}"
        meta = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_message": user_message,
            "assistant_message": assistant_message,
        }
        if metadata:
            meta.update(metadata)

        self.collection.add(
            ids=[doc_id],
            documents=[text],
            metadatas=[meta],
        )
        return doc_id

    def retrieve_relevant(self, query: str, k: int = 5) -> list[str]:
        """Fetch the k most semantically relevant past exchanges for this query."""
        if self.collection.count() == 0:
            return []

        results = self.collection.query(
            query_texts=[query],
            n_results=min(k, self.collection.count()),
        )
        docs = results.get("documents", [[]])[0]
        return docs

    def get_recent(self, limit: int = 10) -> list[dict]:
        """Fetch the most recent memories (useful for a 'recent history' panel)."""
        if self.collection.count() == 0:
            return []
        all_items = self.collection.get()
        combined = list(zip(all_items["ids"], all_items["metadatas"]))
        combined.sort(key=lambda x: x[1].get("timestamp", ""), reverse=True)
        return [m for _, m in combined[:limit]]

    def clear(self):
        """Wipe all memories for this user (for testing / reset button)."""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            embedding_function=self.embed_fn,
        )
