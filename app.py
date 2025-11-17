import os
import io
import sys
import textwrap
import re
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

# Initialize session-based chat history so previous results persist across queries
if "chat_history" not in st.session_state:
    # Each item: {"role": "user"|"assistant", "content": str}
    st.session_state.chat_history = []

# Sidebar controls
with st.sidebar:
    st.subheader("Session")
    if st.button("Clear conversation", use_container_width=True):
        # Clear UI session history
        st.session_state.chat_history = []
        # Remove persisted history to avoid influencing future answers
        try:
            Path(history_file).unlink(missing_ok=True)
        except Exception:
            pass
        # Rerun to reflect cleared state
        try:
            st.experimental_rerun()
        except Exception:
            st.rerun()

# Global CSS to normalize typography and ensure clean, consistent rendering
st.markdown(
    """
    <style>
      /* Normalize overall font and sizes */
      html, body, [class^="css"], .stMarkdown, .stText, .stTextInput, .stTextArea {
        font-family: Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif !important;
      }
      .stMarkdown p, .stMarkdown li, .stMarkdown code, .stMarkdown pre {
        font-size: 1rem !important;
        line-height: 1.6 !important;
        word-wrap: break-word !important;
        overflow-wrap: anywhere !important;
        white-space: pre-wrap !important;
      }
      /* Flatten headings so model-produced ###/#### don't look huge */
      .stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
      .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        margin: 0.4rem 0 0.6rem 0 !important;
      }
      /* Style code blocks slightly */
      .stMarkdown pre, .stMarkdown code {
        background: #f6f8fa !important;
        border-radius: 6px !important;
      }
      .stMarkdown pre { padding: 0.6rem !important; }

      /* Ensure headings and lists inside chat bubbles are left-aligned and consistent */
      .stChatMessage .stMarkdown h1,
      .stChatMessage .stMarkdown h2,
      .stChatMessage .stMarkdown h3,
      .stChatMessage .stMarkdown h4,
      .stChatMessage .stMarkdown h5,
      .stChatMessage .stMarkdown h6 {
        text-align: left !important;
        display: block !important;
        margin: 0.4rem 0 0.4rem 0 !important;
      }
      .stChatMessage .stMarkdown ul,
      .stChatMessage .stMarkdown ol {
        margin: 0.2rem 0 0.6rem 1.2rem !important;
        padding-left: 1.0rem !important;
      }
      .stChatMessage .stMarkdown li {
        margin: 0.1rem 0 !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# Create a container for history so we can render it after processing input while keeping
# the input widget at the bottom of the page visually.
history_container = st.container()

# Using Streamlit chat components places the input docked at the bottom automatically
prompt = st.chat_input(
    "Type your message and press Enter...",
    key="chat_input",
)

# Handle submission from chat input
if prompt and prompt.strip():
    user_text = prompt.strip()
    # Append user message to session history
    st.session_state.chat_history.append({"role": "user", "content": user_text})

    # Capture stdout from run_chat to display in Streamlit
    f = io.StringIO()
    try:
        with redirect_stdout(f):
            # run_chat supports optional history_path
            run_chat(user_text, history_path=Path(history_file))
    except Exception as e:
        st.error(f"Error while running chat: {e}")
    finally:
        output = f.getvalue()

    # Extract answer: drop any timing logs
    lines = [ln for ln in output.splitlines() if ln.strip()]
    non_timing_lines = [ln for ln in lines if not ln.startswith("[TIMING]")]
    final_text = "\n".join(non_timing_lines).strip() if non_timing_lines else ""

    # Normalize markdown to avoid accidental code blocks from leading spaces
    if final_text:
        normalized = textwrap.dedent(final_text).replace("\t", "    ").strip()
    else:
        normalized = ""

    # Strip emojis and control characters to keep output clean
    if normalized:
        try:
            # Remove most emoji code points (supplementary planes)
            normalized = re.sub(r"[\U00010000-\U0010FFFF]", "", normalized)
        except re.error:
            # Fallback for narrow builds: remove common pictographs by range chunks
            normalized = re.sub(r"[\u2600-\u27BF]", "", normalized)
        # Normalize whitespace
        normalized = re.sub(r"[\t\r]+", " ", normalized)
        normalized = re.sub(r"\n{3,}", "\n\n", normalized).strip()

    # Persist the assistant response to the session history
    if normalized:
        st.session_state.chat_history.append({"role": "assistant", "content": normalized})
    else:
        st.session_state.chat_history.append({"role": "assistant", "content": "No answer returned (check logs)."})

# Render conversation history (always show, including immediately after a submission)
with history_container:
    if st.session_state.chat_history:
        for msg in st.session_state.chat_history:
            role = msg.get("role")
            with st.chat_message("user" if role == "user" else "assistant"):
                # Use markdown to allow basic formatting while normalized by our CSS
                st.markdown(msg.get("content", ""))

st.markdown("---")
st.caption("Built with Streamlit")
