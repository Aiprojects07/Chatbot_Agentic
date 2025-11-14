import os
import io
import sys
from pathlib import Path
from contextlib import redirect_stdout
from typing import Any, Dict

import streamlit as st
from dotenv import load_dotenv
try:
    import toml  # type: ignore
except Exception:
    toml = None

PROJECT_DIR = Path(__file__).parent

# Load .env if running locally
load_dotenv(dotenv_path=PROJECT_DIR / ".env", override=False)

# Load local secrets from secrets.toml if present (useful locally). Streamlit secrets will override.
secrets_toml_path = PROJECT_DIR / "secrets.toml"
if secrets_toml_path.exists() and toml is not None:
    try:
        data: Dict[str, Any] = toml.load(str(secrets_toml_path))  # type: ignore[arg-type]
        for k, v in (data or {}).items():
            if isinstance(v, (str, int, float)) and k not in os.environ:
                os.environ[k] = str(v)
    except Exception as e:
        # Non-fatal: continue without local secrets
        pass

# Load secrets (preferred on Streamlit Cloud)
SECRETS_TO_IMPORT = [
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "PINECONE_API_KEY",
    "PINECONE_ENV",
    "PINECONE_ENVIRONMENT",
    "PINECONE_INDEX",
    "PINECONE_INDEX_NAME",
    "PINECONE_NAMESPACE",
    "PINECONE_DIMENSION",
    "HISTORY_MAX_TURNS",
]
try:
    # st.secrets may raise if no secrets are configured; handle gracefully
    secrets_dict = dict(st.secrets)
except Exception:
    secrets_dict = {}

for key in SECRETS_TO_IMPORT:
    if key in secrets_dict:
        os.environ[key] = str(secrets_dict[key])

# Import after env is ready
from main import run_chat  # noqa: E402

st.set_page_config(page_title="Skills Chatbot", page_icon="💄", layout="wide")

st.title("💄 Skills Chatbot")
st.caption("Ask about products, find similar or complementary items.")

# Use a default history file path internally (no UI exposure)
history_file = str(Path(__file__).parent / "conversation_history.jsonl")

user_query = st.text_area("Your message", height=120, placeholder="e.g., Which others products I can use with Typsy Beauty Cocoa Peptide Velvet Lipstick Brownie Bite Medium 02 lipstick?")

run_clicked = st.button("Ask")

if run_clicked and user_query.strip():
    # Capture stdout from run_chat to display in Streamlit
    f = io.StringIO()
    try:
        with redirect_stdout(f):
            # run_chat supports optional history_path
            run_chat(user_query, history_path=Path(history_file))
    except Exception as e:
        st.error(f"Error while running chat: {e}")
    finally:
        output = f.getvalue()

    # Extract answer: drop any timing logs
    lines = [ln for ln in output.splitlines() if ln.strip()]
    non_timing_lines = [ln for ln in lines if not ln.startswith("[TIMING]")]
    final_text = "\n".join(non_timing_lines).strip() if non_timing_lines else ""

    if final_text:
        st.subheader("Answer")
        st.write(final_text)
    else:
        st.info("No answer returned (check logs).")

st.markdown("---")
st.caption("Built with Streamlit")
