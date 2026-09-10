"""
Lightweight RAG engine for the IEEE RAS Assistant.

Design choices (kept deliberately simple so it's easy to explain and cheap
to run on a free-tier deployment):

- Retrieval: TF-IDF + cosine similarity over locally stored knowledge-base
  files. No external embedding API calls are needed for retrieval, which
  keeps the app fast, free, and independent of any single provider's
  embedding quota.
- Generation: Groq's free, OpenAI-compatible API (Llama 3.3 70B by default)
  is used only for the final answer, conditioned on the retrieved chunks.
  Groq's free tier needs no credit card and uses a standard API key, which
  sidesteps the compatibility issues some providers' newer key formats have
  had with certain SDKs.
- Knowledge base: any .md/.txt file placed in knowledge_base/ is picked up
  automatically and chunked by paragraph/heading, so the corpus can be
  extended without touching any code.
"""

from __future__ import annotations

import os
import re
import glob
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from openai import OpenAI


KB_DIR = os.path.join(os.path.dirname(__file__), "knowledge_base")
CHUNK_MAX_CHARS = 900
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "llama-3.3-70b-versatile"

SYSTEM_INSTRUCTIONS = """You are the official assistant for the IEEE RAS \
(Robotics and Automation Society) Student Branch Chapter at VIT Chennai.

Rules:
- Answer ONLY using the information given in the "Context" section below.
- If the answer is not contained in the context, say you don't have that \
information yet and suggest the person check the chapter's official \
Instagram/LinkedIn or contact the chapter directly. Do not make anything up.
- Be concise, friendly, and helpful — you are talking to students who might \
want to join or learn about the chapter.
- When useful, mention which part of the context you're drawing from (e.g. \
"According to the chapter's founding info...").
"""


@dataclass
class Chunk:
    text: str
    source: str


def _split_into_chunks(text: str, source: str) -> list[Chunk]:
    """Split a document into paragraph-ish chunks capped at CHUNK_MAX_CHARS."""
    # Split on blank lines / markdown headings first.
    raw_parts = re.split(r"\n\s*\n", text.strip())
    chunks: list[Chunk] = []
    buffer = ""
    for part in raw_parts:
        part = part.strip()
        if not part:
            continue
        if len(buffer) + len(part) + 1 <= CHUNK_MAX_CHARS:
            buffer = (buffer + "\n" + part).strip()
        else:
            if buffer:
                chunks.append(Chunk(text=buffer, source=source))
            # If a single part is itself too long, hard-split it.
            if len(part) > CHUNK_MAX_CHARS:
                for i in range(0, len(part), CHUNK_MAX_CHARS):
                    chunks.append(Chunk(text=part[i : i + CHUNK_MAX_CHARS], source=source))
                buffer = ""
            else:
                buffer = part
    if buffer:
        chunks.append(Chunk(text=buffer, source=source))
    return chunks


def load_knowledge_base(kb_dir: str = KB_DIR) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(glob.glob(os.path.join(kb_dir, "*"))):
        if not path.lower().endswith((".md", ".txt")):
            continue
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        source = os.path.basename(path)
        chunks.extend(_split_into_chunks(text, source))
    return chunks


class RagIndex:
    """A tiny in-memory TF-IDF index over the knowledge base chunks."""

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(stop_words="english")
        corpus = [c.text for c in chunks] if chunks else [""]
        self.matrix = self.vectorizer.fit_transform(corpus)

    def retrieve(self, query: str, top_k: int = 4) -> list[Chunk]:
        if not self.chunks:
            return []
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.matrix)[0]
        top_idx = np.argsort(sims)[::-1][:top_k]
        return [self.chunks[i] for i in top_idx if sims[i] > 0]


def build_index() -> RagIndex:
    return RagIndex(load_knowledge_base())


_client: OpenAI | None = None


def configure_llm(api_key: str) -> None:
    global _client
    _client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)


def answer_question(
    question: str,
    index: RagIndex,
    chat_history: list[dict] | None = None,
    top_k: int = 4,
) -> tuple[str, list[Chunk]]:
    """Retrieve relevant chunks and ask the LLM to answer using them.

    Returns (answer_text, retrieved_chunks) so the UI can show sources.
    """
    retrieved = index.retrieve(question, top_k=top_k)

    if retrieved:
        context_block = "\n\n---\n\n".join(
            f"[Source: {c.source}]\n{c.text}" for c in retrieved
        )
    else:
        context_block = "(no relevant context found in the knowledge base)"

    messages = [{"role": "system", "content": f"{SYSTEM_INSTRUCTIONS}\n\nContext:\n{context_block}"}]
    if chat_history:
        # chat_history's last entry is this same question (the caller already
        # appended it before calling us) - exclude it here since it's added
        # explicitly below, then keep a handful of prior turns for context.
        prior_turns = chat_history[:-1][-6:]
        for turn in prior_turns:
            role = "assistant" if turn["role"] == "assistant" else "user"
            messages.append({"role": role, "content": turn["content"]})
    messages.append({"role": "user", "content": question})

    if _client is None:
        raise RuntimeError("LLM client not configured. Call configure_llm(api_key) first.")

    response = _client.chat.completions.create(model=GROQ_MODEL, messages=messages)
    return response.choices[0].message.content, retrieved
