# history_logic.py

import os
import json
import logging
from typing import List, Dict

DEFAULT_HISTORY_MAX_TURNS = 3  # fallback if env var not provided


def get_history_max_turns() -> int:
    """Return the history window size. Allows env override but centralizes default."""
    try:
        return int(os.getenv("HISTORY_MAX_TURNS", str(DEFAULT_HISTORY_MAX_TURNS)))
    except Exception:
        return DEFAULT_HISTORY_MAX_TURNS


def load_conv_history(path: str, max_turns: int) -> List[Dict[str, str]]:
    """Load last max_turns user+assistant pairs from a JSONL file as Anthropic messages.
    Each JSONL line should be an object: {"role": "user"|"assistant", "content": str}.
    Returns a list of messages usable directly in Anthropic's messages.create.
    """
    history: List[Dict[str, str]] = []
    if not path or not os.path.isfile(path):
        return history
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        # Keep only the last 2*max_turns lines (user+assistant per turn)
        if max_turns > 0:
            lines = lines[-(max_turns * 2):]
        for ln in lines:
            try:
                obj = json.loads(ln)
                role = obj.get("role")
                content = obj.get("content")
                if role in ("user", "assistant") and isinstance(content, str):
                    history.append({"role": role, "content": content})
            except Exception:
                # ignore malformed lines
                continue
    except Exception:
        # Non-fatal: failure to load history should not crash the app
        logging.debug("Failed to load history file: %s", path)
    return history


def append_conv_history(path: str, role: str, content: str) -> None:
    """Append a single message to the JSONL history file, creating directories as needed."""
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"role": role, "content": content}, ensure_ascii=False) + "\n")
    except Exception:
        # Non-fatal: don't block the chat loop
        logging.debug("Failed to append to history file: %s", path)