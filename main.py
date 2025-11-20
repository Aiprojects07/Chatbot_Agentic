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
    general_product_qna,
)
# history_logic is not needed in router-only mode

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

tools = [
    {
        "name": "resolve_product",
        "description": """Use this tool ONLY when a downstream step requires a resolved product vector (SKU/context) — for example:
        • Finding similar products (retrieve_similar_products)
        • Recommending complementary 'use-with' products (retrieve_use_with_products)

        Do NOT use this tool to answer simple factual product questions (ingredients, finish, usage, opinions). For those, call general_product_qna directly.

        How to use:
        - Pass the exact product name (as written by the user if possible). Optionally pass category to narrow the search. Use top_k >= 8.
        - After resolving, proceed with the required downstream tool.""",
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
      
      Do NOT generate or transform the query here; downstream functions will handle query refinement. Provide only the inputs specified by the schema.

      Category parameter guidance:
      - Set the 'category' only if the user explicitly specifies it OR you are highly confident from context.
      - Otherwise omit it; the tool will infer complementary categories downstream.

      Prerequisite when a SPECIFIC product name is mentioned:
      - FIRST call resolve_product to get the exact SKU and prior resolved products.
      - THEN call retrieve_use_with_products, passing selected_sku and selected_products from resolve_product.

      Examples:
      - "Which other products can I use with Typsy Beauty Cocoa Peptide Velvet Lipstick Brownie Bite Medium 02?"
        → resolve_product (to get SKU) → retrieve_use_with_products (with selected_sku, selected_products).""",
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
    ,
    {
      "name": "general_product_qna",
      "description": """Router guidance only.

      Single rule:
      - General Product QnA handles any factual question that does NOT require comparing one product to others.
        (Factual includes: ingredients, finish, shade suitability, texture, performance, usage, brand notes, quick facts.)

      Do NOT call resolve_product before this tool.

      Category parameter guidance:
      - If confident, set 'category' to one of [lipstick, lip_balm_treatment, lip_liner]. Otherwise omit it and search broadly.

      Top-K guidance:
      - Always set top_k: 5 for this tool (fetch 5 candidates and answer from them). Do not pass larger values.""",
      "input_schema": {
        "type": "object",
        "properties": {
          "query": {"type": "string", "description": "User's natural-language question"},
          "category": {"type": "string", "enum": ["lipstick", "lip_balm_treatment", "lip_liner"], "description": "Category to search within"},
          "top_k": {"type": "integer", "minimum": 5, "description": "How many candidates to fetch before answering"}
        },
        "required": ["query"],
        "additionalProperties": False
      }
    }
]

TOOL_FUNCS = {
    "retrieve_similar_products": retrieve_similar_products,
    "retrieve_use_with_products": retrieve_use_with_products,
    "resolve_product": resolve_product,
    "general_product_qna": general_product_qna,
}

def run_chat(user_message: str, history_path: Path = None) -> None:
    # Router-with-chaining: the LLM decides which tools to call and may chain multiple calls.
    temperature = 0.1
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
        for tu in tool_uses:
            tool_use_id = getattr(tu, "id", None)
            name = getattr(tu, "name", None)
            arguments = getattr(tu, "input", {}) or {}

            # Auto-augment retrieve_use_with_products with resolved context if LLM omitted it
            if name == "retrieve_use_with_products":
                if isinstance(arguments, dict):
                    if "selected_products" not in arguments and "selected_products" in resolved_context:
                        arguments["selected_products"] = resolved_context.get("selected_products")
                    if "selected_sku" not in arguments and "selected_sku" in resolved_context:
                        arguments["selected_sku"] = resolved_context.get("selected_sku")
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
            if name in ("retrieve_use_with_products", "retrieve_similar_products", "general_product_qna") and not handled_selected_payload:
                print(content_to_send)
                total_elapsed = time.perf_counter() - total_start
                print(f"[TIMING] Total run_chat execution took {total_elapsed:.3f}s")
                return

            # Minimize cost: if resolve_product was handled (we extracted selected_* into resolved_context),
            # do NOT echo the full heavy JSON back to the router LLM. Instead, send a compact ack summary.
            if name == "resolve_product" and handled_selected_payload:
                compact_ack = {
                    "selected_sku": resolved_context.get("selected_sku"),
                    "selected_products_count": len(resolved_context.get("selected_products") or []),
                }
                tool_results_payload.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": json.dumps(compact_ack),
                })
            else:
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
    # run_chat("Which 5 lipsticks are similar to Typsy Beauty Cocoa Peptide Velvet Lipstick Brownie Bite Medium 02?")
    # run_chat("Which others products I can use it with Typsy Beauty Cocoa Peptide Velvet Lipstick Brownie Bite Medium 02 lipstick?")
    # run_chat("How does this Typsy Beauty Cocoa Peptide Velvet Lipstick Brownie Bite Medium 02 looks on indian skin tone?")
    # run_chat("What ingreadient use in Dr. PawPaw Lip & Eye Balm?")
    # run_chat(" Show me top 5 lip balm which is suitable on dry lips?")
    # run_chat("Show me top 5 lipsticks which is suitable for wedding purpose?")
    run_chat("Which others products I can use with Dr. PawPaw Lip & Eye Balm??")