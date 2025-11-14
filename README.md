# Skills Chatbot

A Streamlit-based chatbot that uses Anthropic for chat + function calling, Pinecone for vector search, and OpenAI for embeddings. It supports conversation history and two retrieval tools:

- `retrieve_similar_products`
- `retrieve_use_with_products`

The app includes precise timing logs internally and maintains a JSONL history file (not exposed in the UI).

## Requirements

- Python 3.9+
- A Pinecone index compatible with your embedding dimensions
- API keys:
  - `ANTHROPIC_API_KEY`
  - `OPENAI_API_KEY`
  - `PINECONE_API_KEY`
  - `PINECONE_INDEX` or `PINECONE_INDEX_NAME`
  - `PINECONE_ENV` or `PINECONE_ENVIRONMENT`
  - Optional: `PINECONE_NAMESPACE`, `PINECONE_DIMENSION`, `HISTORY_MAX_TURNS`

## Local Setup

1. Create and activate a virtual environment:

```
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Provide secrets (choose one):

- Create a `.env` file in the project root with your keys
- Or create a `secrets.toml` in the project root (gitignored)
- Optional: Streamlit native `.streamlit/secrets.toml` (gitignored)

4. Run the app:

```
streamlit run app.py
```

If `streamlit` is not in PATH:

```
./venv/bin/streamlit run app.py
```

## Files

- `app.py` — Streamlit UI
- `main.py` — Chat loop, tools wiring, history usage, timing logs
- `tools/product_tools.py` — Tool implementations using Pinecone + OpenAI embeddings
- `history_logic.py` — Conversation history utilities (JSONL)
- `requirements.txt` — Dependencies
- `.gitignore` — Excludes venv, secrets, cache, and history file

## Streamlit Cloud Deployment

1. Push this repo to GitHub.
2. On Streamlit Cloud, create a new app pointing to `app.py`.
3. Set Secrets in the app settings (recommended):
   - `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `PINECONE_API_KEY`
   - `PINECONE_INDEX` (or `PINECONE_INDEX_NAME`), `PINECONE_ENV` (or `PINECONE_ENVIRONMENT`)
   - Optional: `PINECONE_NAMESPACE`, `PINECONE_DIMENSION`, `HISTORY_MAX_TURNS`

The app will automatically read Streamlit secrets and override local `.env`/`secrets.toml`.

## Notes

- History file: defaults to `conversation_history.jsonl` in the project root.
- Timing logs: visible in server stdout (not rendered in the UI).
- Ensure your Pinecone index dimension matches the chosen OpenAI embedding model (default: `text-embedding-3-large`, 3072 dims).
