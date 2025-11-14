# main.py

import os
from anthropic import Anthropic
from typing import Dict, Any
import json
from pathlib import Path
import time
from dotenv import load_dotenv

# Ensure environment variables from .env are loaded (project root) BEFORE importing tools
load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=False)

# Import local tool functions for execution (these read env at import time)
from tools.product_tools import (
    retrieve_similar_products,
    retrieve_use_with_products,
    resolve_product,
)
from history_logic import (
    get_history_max_turns,
    load_conv_history,
    append_conv_history,
)

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

tools = [
    {
        "name": "resolve_product",
        "description": """Use this tool whenever the user's query mentions a SPECIFIC product name and you need reliable product context (id/metadata) to answer.
        
        Use this tool for:
        - Any question that references a specific, named product (e.g., shade suitability, finish, longevity, comfort, transfer, ingredients, usage tips)
        - Preparing to find similar products (then call retrieve_similar_products)
        - Preparing to find complementary products (then call retrieve_use_with_products)
        
        How to use:
        - Pass the exact product name (as written by the user if possible). Optionally pass category to narrow the search. Use top_k >= 10.
        - After resolving, synthesize the final answer using the retrieved context (you may then call another tool if needed).""",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Exact product name to search for"
                },
                "category": {
                    "type": "string",
                    "enum": ["lipstick", "lip_balm_treatment", "lip_liner"],
                    "description": "Optional category to filter by (lipstick, lip_balm_treatment, or lip_liner)"
                },
                "top_k": {
                    "type": "integer",
                    "minimum": 10,
                    "description": "Number of candidates to return"
                }
            },
            "required": ["name", "top_k"],
            "additionalProperties": False
      }
    },
    {
        "name": "retrieve_similar_products",
        "description": """Use this tool to find products that are similar to a given product description.
        
        When to use this tool:
        - When the user asks for products similar to a specific item (e.g., 'find me products like this lipstick')
        - When you need to recommend alternatives to a specific product
        - When you want to show variations of a particular product
      
      How to use:
      1. Provide a detailed product description including key features, color, and type
      2. Set top_k to control the number of similar products to return (minimum 10)
      
      Example usage:
      - 'Find 5 lipsticks similar to a long-lasting matte red lipstick'
      - 'Show me 3 lip balms like a medicated, unscented lip treatment'""",
      "input_schema": {
        "type": "object",
        "properties": {
          "product_description": {"type": "string", "description": "Detailed description of the reference product including key features"},
          "category": {"type": "string", "enum": ["lipstick", "lip_balm_treatment", "lip_liner"], "description": "Optional category to filter by (lipstick, lip_balm_treatment, or lip_liner)"},
          "top_k": {"type": "integer", "minimum": 10, "description": "Number of similar products to return"}
        },
        "required": ["product_description", "top_k"],
        "additionalProperties": False
      }
    },
    {
      "name": "retrieve_use_with_products",
      "description": """Use this tool to find complementary products based on a description.
      
      When to use this tool:
      - When the user asks for products that would complement a specific type of product
      - When suggesting product combinations or complete routines
      - When recommending items that work well together
      
      How to use:
      1. Create a descriptive query about the type of complementary products to find
      2. Specify the target category (lipstick, lip_balm_treatment, or lip_liner)
      3. Set top_k to control the number of results (minimum 10)
      
      Example usage:
      - 'Find hydrating lip balms that work well with matte lipsticks' with category 'lip_balm_treatment'
      - 'Show me lip liners that complement brown-berry neutral lipstick shades' with category 'lip_liner'""",
      "input_schema": {
        "type": "object",
        "properties": {
          "query": {"type": "string", "description": "Description of complementary products to find"},
          "category": {"type": "string", "enum": ["lipstick", "lip_balm_treatment", "lip_liner"], "description": "Category to search within"},
          "top_k": {"type": "integer", "minimum": 10, "description": "Number of complementary products to return"}
        },
        "required": ["query", "category", "top_k"],
        "additionalProperties": False
      }
    }
]

TOOL_FUNCS = {
    "retrieve_similar_products": retrieve_similar_products,
    "retrieve_use_with_products": retrieve_use_with_products,
    "resolve_product": resolve_product,
}

def run_chat(user_message: str, history_path: Path = None) -> None:
    # Load prompt template and generation settings
    prompt_path = Path(__file__).parent / "prompt1.txt"
    system_text = None
    temperature = 0.7
    max_tokens = 2000
    # History path
    history_path = history_path or (Path(__file__).parent / "conversation_history.jsonl")
    if prompt_path.exists():
        try:
            system_text = prompt_path.read_text()
        except Exception:
            # Fall back to defaults on file read error
            system_text = None

    # Load prior history and add current user message
    messages = load_conv_history(str(history_path), get_history_max_turns())
    messages.append({"role": "user", "content": user_message})

    # Loop to allow multi-step tool use (e.g., resolve -> query search -> finalize)
    total_start = time.perf_counter()
    for _ in range(5):  # safety cap to avoid infinite loops
        llm_start = time.perf_counter()
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            system=system_text,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=messages,
            tools=tools,
        )
        llm_elapsed = time.perf_counter() - llm_start
        print(f"[TIMING] LLM call took {llm_elapsed:.3f}s")

        content_blocks = response.content
        tool_uses = [b for b in content_blocks if getattr(b, "type", None) == "tool_use"]

        if not tool_uses:
            # No tool calls: final answer (print only text blocks)
            final_text = "\n\n".join([
                getattr(b, "text", "") for b in content_blocks if getattr(b, "type", None) == "text"
            ])
            print(final_text or content_blocks)
            # Append to history
            append_conv_history(str(history_path), "user", user_message)
            append_conv_history(str(history_path), "assistant", final_text)
            total_elapsed = time.perf_counter() - total_start
            print(f"[TIMING] Total run_chat execution took {total_elapsed:.3f}s")
            return

        # Execute requested tools and append results
        tool_results_payload = []
        for block in tool_uses:
            name = getattr(block, "name", None)
            tool_use_id = getattr(block, "id", None)
            arguments: Dict[str, Any] = getattr(block, "input", {}) or {}
            func = TOOL_FUNCS.get(name)
            if not func:
                tool_results_payload.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": f"Unknown tool: {name}"
                })
                continue
            try:
                tool_start = time.perf_counter()
                result_str = func(**arguments)
                tool_elapsed = time.perf_counter() - tool_start
                print(f"[TIMING] Tool '{name}' executed in {tool_elapsed:.3f}s with args={arguments}")
            except Exception as e:
                result_str = os.linesep.join([
                    "{\"error\": \"tool execution failed\"}",
                    f"details: {e}",
                ])
                tool_elapsed = time.perf_counter() - tool_start if 'tool_start' in locals() else 0.0
                print(f"[TIMING] Tool '{name}' failed after {tool_elapsed:.3f}s; error={e}")
            tool_results_payload.append({
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": result_str,
            })

        # Provide the assistant's tool request and our tool results back in the conversation
        messages.extend([
            {"role": "assistant", "content": content_blocks},
            {"role": "user", "content": tool_results_payload},
        ])

    # If we exit the loop without producing a final answer, log total time and exit
    total_elapsed = time.perf_counter() - total_start
    print(f"[TIMING] Total run_chat execution took {total_elapsed:.3f}s (max steps reached without final answer)")
    return


if __name__ == "__main__":
    # Example: user asks a natural question; model chooses the right tool
    # run_chat("Which 5 lipsticks are similar to Typsy Beauty Cocoa Peptide Velvet Lipstick Brownie Bite Medium 02?")
    run_chat("Which others products I can use it with Typsy Beauty Cocoa Peptide Velvet Lipstick Brownie Bite Medium 02 lipstick?")
    # run_chat("How does this Typsy Beauty Cocoa Peptide Velvet Lipstick Brownie Bite Medium 02 looks on indian skin tone?")
    # run_chat(" What ingreadient use in Dr. PawPaw Lip & Eye Balm?")
    # run_chat(" Show me top 5 lip balm which is suitable on dry lips?")
    # run_chat("Show me top 5 lipsticks which is suitable for wedding purpose?")