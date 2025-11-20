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
# history_logic is not needed in router-only mode

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
        - Pass the exact product name (as written by the user if possible). Optionally pass category to narrow the search. Use top_k >= 8.
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
        "description": """Router guidance only: Call this tool when the user asks for products similar to another product within the lip domain (e.g., alternatives, lookalikes, or variations).
        
        When to call:
        - User requests similar/alternative items to a described or known product
        - User wants variations of a particular product
        
        Do NOT generate or transform the query here; downstream functions will handle query refinement. Provide only the inputs specified by the schema.""",
        "input_schema": {
          "type": "object",
          "properties": {
            "product_name": {"type": "string", "description": "Complete product name to anchor similarity (e.g., brand + line + shade)"},
            "category": {"type": "string", "enum": ["lipstick", "lip_balm_treatment", "lip_liner"], "description": "Optional category to filter by (lipstick, lip_balm_treatment, or lip_liner)"},
            "top_k": {"type": "integer", "minimum": 10, "description": "Number of similar products to return"},
            "selected_sku": {"type": "string", "description": "Optional SKU resolved from a prior step (e.g., resolve_product)"},
            "selected_products": {"type": "array", "description": "Optional prior resolved products array from resolve_product to improve similarity selection"}
          },
          "required": ["product_name", "top_k"],
          "additionalProperties": False
        }
    },
    {
      "name": "retrieve_use_with_products",
      "description": """Router guidance only: Call this tool when the user asks for complementary products (items to use together) within the lip domain.
      
      When to call:
      - User requests products that pair well with a specified product or product type
      - Building product combinations or complete lip routines
      - Recommending items that work well together
      
      Do NOT generate or transform the query here; downstream functions will handle query refinement. Provide only the inputs specified by the schema.""",
      "input_schema": {
        "type": "object",
        "properties": {
          "product_name": {"type": "string", "description": "Complete product name for which to find complementary items"},
          "category": {"type": "string", "enum": ["lipstick", "lip_balm_treatment", "lip_liner"], "description": "Category to search within"},
          "top_k": {"type": "integer", "minimum": 5, "description": "Number of complementary products to return"},
          "selected_sku": {"type": "string", "description": "Optional SKU resolved from a prior step (e.g., resolve_product)"},
          "selected_products": {"type": "array", "description": "Optional prior resolved products array from resolve_product to improve complement selection"}
        },
        "required": ["product_name", "category", "top_k"],
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
    # Router-with-chaining: the LLM decides which tools to call and may chain multiple calls.
    temperature = 0.2
    max_tokens = 1024
    messages = [{"role": "user", "content": user_message}]

    total_start = time.perf_counter()
    # Persist resolved context across steps so downstream tools can leverage it
    resolved_context: Dict[str, Any] = {}
    for step in range(6):  # safety cap
        llm_start = time.perf_counter()
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            temperature=temperature,
            max_tokens=max_tokens,
            messages=messages,
            tools=tools,
        )
        llm_elapsed = time.perf_counter() - llm_start
        print(f"[TIMING] LLM call took {llm_elapsed:.3f}s (step {step+1})")

        content_blocks = response.content
        tool_uses = [b for b in content_blocks if getattr(b, "type", None) == "tool_use"]

        if not tool_uses:
            # Final answer: print only text blocks and exit
            final_text = "\n\n".join([
                getattr(b, "text", "") for b in content_blocks if getattr(b, "type", None) == "text"
            ])
            print(final_text or content_blocks)
            total_elapsed = time.perf_counter() - total_start
            print(f"[TIMING] Total run_chat execution took {total_elapsed:.3f}s")
            return

        # Execute requested tools (all in this step) and append results
        tool_results_payload = []
        for block in tool_uses:
            name = getattr(block, "name", None)
            tool_use_id = getattr(block, "id", None)
            arguments: Dict[str, Any] = getattr(block, "input", {}) or {}
            # Generalized context threading: inject any keys present in resolved_context
            # that are also declared in the tool's input_schema properties (to respect
            # additionalProperties=False), but only if they are not already provided.
            try:
                tool_schema = next((t for t in tools if t.get("name") == name), None)
                schema_props = set(((tool_schema or {}).get("input_schema") or {}).get("properties", {}).keys())
                for k, v in resolved_context.items():
                    if k in schema_props and k not in arguments and v is not None:
                        arguments[k] = v
            except Exception:
                # If schema lookup fails, skip injection safely
                pass
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

            # Filter tool result to only include selected_products (and selected_sku) when available
            content_to_send = result_str
            handled_selected_payload = False
            try:
                parsed = json.loads(result_str)
                if isinstance(parsed, dict) and "selected_products" in parsed:
                    filtered_payload = {
                        "selected_products": parsed.get("selected_products"),
                        "selected_sku": parsed.get("selected_sku"),
                    }
                    content_to_send = json.dumps(filtered_payload)
                    # Update resolved context for subsequent tools
                    resolved_context.update({
                        k: v for k, v in filtered_payload.items() if v is not None
                    })
                    handled_selected_payload = True
            except Exception:
                pass

            # If this tool is a final-packaging style and we did not process a selected_products payload,
            # treat its output as the final answer (handles plain-text responses too).
            if name in ("retrieve_use_with_products", "retrieve_similar_products") and not handled_selected_payload:
                print(content_to_send)
                total_elapsed = time.perf_counter() - total_start
                print(f"[TIMING] Total run_chat execution took {total_elapsed:.3f}s")
                return

            tool_results_payload.append({
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": content_to_send,
            })

        # Feed the assistant tool requests and our tool results back into the conversation
        messages.extend([
            {"role": "assistant", "content": content_blocks},
            {"role": "user", "content": tool_results_payload},
        ])

    total_elapsed = time.perf_counter() - total_start
    print(f"[TIMING] Total run_chat execution took {total_elapsed:.3f}s (max steps reached without final answer)")
    return


if __name__ == "__main__":
    # Example: user asks a natural question; model chooses the right tool
    run_chat("Which 5 lipsticks are similar to Typsy Beauty Cocoa Peptide Velvet Lipstick Brownie Bite Medium 02?")
    # run_chat("Which others products I can use it with Typsy Beauty Cocoa Peptide Velvet Lipstick Brownie Bite Medium 02 lipstick?")
    # run_chat("How does this Typsy Beauty Cocoa Peptide Velvet Lipstick Brownie Bite Medium 02 looks on indian skin tone?")
    # run_chat(" What ingreadient use in Dr. PawPaw Lip & Eye Balm?")
    # run_chat(" Show me top 5 lip balm which is suitable on dry lips?")
    # run_chat("Show me top 5 lipsticks which is suitable for wedding purpose?")