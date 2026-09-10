# IEEE RAS Assistant (RAG-based) — VIT Chennai Student Chapter

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about
the IEEE Robotics and Automation Society (RAS), with a focus on the VIT
Chennai Student Branch Chapter, using publicly available information.

## How it works (RAG pipeline)

1. **Knowledge base** (`knowledge_base/*.md`): plain-text files containing
   publicly sourced facts about IEEE RAS globally and the VIT Chennai
   chapter (founding/inauguration details, mission, how to join, etc.).
2. **Chunking**: each file is split into paragraph-sized chunks at app
   startup (`rag_engine.load_knowledge_base`).
3. **Retrieval**: chunks are indexed with a TF-IDF vectorizer
   (`scikit-learn`). When a user asks a question, the question is vectorized
   and compared against all chunks with cosine similarity; the top-k most
   relevant chunks are retrieved (`RagIndex.retrieve`).
4. **Generation**: the retrieved chunks are inserted into a prompt, along
   with the recent conversation history, and sent to **Llama 3.3 70B via
   Groq's free, OpenAI-compatible API** to produce a grounded answer
   (`rag_engine.answer_question`). The model is explicitly instructed to
   only answer from the given context and to say when it doesn't know.
5. **UI**: a Streamlit chat interface (`app.py`) ties it together, shows
   which source file(s) were used for each answer, and lets the user reset
   the conversation.

This keeps the whole pipeline free to run: TF-IDF retrieval needs no paid
API calls, and Groq's free tier is enough for a project like this.

## Project structure
```
ieee-ras-rag-bot/
├── app.py                  # Streamlit UI
├── rag_engine.py           # Chunking, TF-IDF retrieval, Gemini generation
├── requirements.txt
├── knowledge_base/
│   ├── ieee_ras_global.md
│   ├── vit_chennai_ras_chapter.md
│   └── joining_and_membership.md
└── .streamlit/
    └── secrets.toml.example
```

## Run it locally
```bash
git clone <your-repo-url>
cd ieee-ras-rag-bot
pip install -r requirements.txt

# Get a free Groq API key: https://console.groq.com/keys
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# then edit .streamlit/secrets.toml and paste your key

streamlit run app.py
```
(Alternatively, just paste the API key into the sidebar text box when the
app opens — nothing is stored anywhere.)

## Deploy it for free (Streamlit Community Cloud)

1. **Push this folder to a new GitHub repo** (public or private):
   ```bash
   cd ieee-ras-rag-bot
   git init
   git add .
   git commit -m "IEEE RAS RAG assistant"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<repo-name>.git
   git push -u origin main
   ```
2. Get a **free Groq API key** at https://console.groq.com/keys
   (sign in with any account, no credit card required).
3. Go to **https://share.streamlit.io** → sign in with GitHub → **"New app"**.
4. Pick your repo, branch `main`, and main file path `app.py`.
5. Before clicking Deploy, open **"Advanced settings" → Secrets** and paste:
   ```toml
   GROQ_API_KEY = "paste-your-key-here"
   ```
6. Click **Deploy**. In a minute or two you'll get a public URL like
   `https://<something>.streamlit.app` — that's the link to submit.

## Extending the knowledge base
Drop any additional `.md` or `.txt` file into `knowledge_base/` — e.g.
exported Instagram/LinkedIn captions, an event brochure, or a list of
current office bearers — and it will automatically be picked up the next
time the app restarts (or redeploys). No code changes needed.

## Notes on the current knowledge base
Content was gathered from publicly available IEEE sources: the IEEE RAS
global society's field of interest and activities, and public records of
the VIT Chennai IEEE RAS chapter's inauguration (9 October 2024, jointly
with IEEE Day 2024, IEEE Region 10). Because most of a student chapter's
day-to-day activity lives on social media rather than an indexable website,
`vit_chennai_ras_chapter.md` notes this gap and invites adding more
chapter-specific content over time.
