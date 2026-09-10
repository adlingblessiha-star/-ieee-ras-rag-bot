import os

import streamlit as st

from rag_engine import build_index, configure_llm, answer_question

st.set_page_config(
    page_title="IEEE RAS VIT Chennai — Assistant",
    page_icon="🤖",
    layout="centered",
)

# ---------- API key handling ----------
# Prefer Streamlit secrets (used in deployment); fall back to an env var for
# local development.
api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    if not api_key:
        api_key = st.text_input("Groq API key", type="password", help=(
            "Get a free key at https://console.groq.com/keys (no credit "
            "card needed). For a deployed app, set this as a Streamlit "
            "secret named GROQ_API_KEY instead of typing it here."
        ))
    st.markdown("---")
    st.markdown(
        "**About**\n\n"
        "This is a RAG (Retrieval-Augmented Generation) assistant for the "
        "IEEE RAS Student Branch Chapter at VIT Chennai. It answers using "
        "publicly available information stored in `knowledge_base/`, "
        "retrieved with TF-IDF similarity and answered by Llama 3.3 70B "
        "via Groq's free API."
    )
    if st.button("🔄 Reset conversation"):
        st.session_state.messages = []
        st.rerun()

st.title("🤖 IEEE RAS Assistant")
st.caption("Ask me about IEEE RAS, the VIT Chennai student chapter, or how to get involved.")

if not api_key:
    st.info("Enter a Groq API key in the sidebar to start chatting.")
    st.stop()

configure_llm(api_key)


# ---------- Build / cache the retrieval index ----------
@st.cache_resource(show_spinner="Loading knowledge base...")
def get_index():
    return build_index()


index = get_index()

if not index.chunks:
    st.warning(
        "No knowledge base files found in `knowledge_base/`. Add .md or .txt "
        "files there and restart the app."
    )

# ---------- Chat state ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask about IEEE RAS VIT Chennai...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer, sources = answer_question(
                    user_input, index, chat_history=st.session_state.messages
                )
            except Exception as e:
                answer = (
                    "Sorry, I hit an error calling the LLM. Double-check your "
                    f"API key and quota.\n\nDetails: `{e}`"
                )
                sources = []
        st.markdown(answer)
        if sources:
            with st.expander("📎 Sources used"):
                for c in sources:
                    st.markdown(f"**{c.source}**")
                    st.caption(c.text[:300] + ("..." if len(c.text) > 300 else ""))

    st.session_state.messages.append({"role": "assistant", "content": answer})
