# tools/product_tools.py

import os
import json
from typing import Optional, List, Dict
from pathlib import Path
from anthropic import Anthropic
from openai import OpenAI
try:
    # Optional decorator; not required for runtime since main.py defines tool schemas explicitly
    from anthropic import beta_tool  # type: ignore
except Exception:
    def beta_tool(*args, **kwargs):  # no-op fallback
        def _decorator(func):
            return func
        return _decorator
from pinecone import Pinecone
import requests

# Initialize Pinecone using flexible env var names
_pc_api_key = os.getenv("PINECONE_API_KEY")
_pc_env = os.getenv("PINECONE_ENV") or os.getenv("PINECONE_ENVIRONMENT")
_pc_index_name = os.getenv("PINECONE_INDEX") or os.getenv("PINECONE_INDEX_NAME")
_pc_namespace = os.getenv("PINECONE_NAMESPACE") or None
_pc_dim_env = os.getenv("PINECONE_DIMENSION")
_pc_expected_dim = int(_pc_dim_env) if _pc_dim_env and _pc_dim_env.isdigit() else None

# Create Pinecone client and target index (assumes index already exists)
pc = Pinecone(api_key=_pc_api_key)
index = pc.Index(_pc_index_name)

# LLM client for in-tool disambiguation
_anthropic_client: Optional[Anthropic] = None
try:
    _anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
except Exception:
    _anthropic_client = None

def _load_prompt_text(filename: str) -> Optional[str]:
    """Load prompt text from the project root.

    tools/product_tools.py is inside tools/, so prompts are expected one level up.
    """
    try:
        root = Path(__file__).parent.parent
        path = root / filename
        if path.exists():
            return path.read_text().strip()
    except Exception:
        return None
    return None

def _refine_query_with_llm(base_prompt_filename: str, user_query: str, model_env_var: str) -> str:
    """Use Anthropic to turn user_query into a concise, optimized search query based on a prompt file.

    Falls back to the original user_query if Anthropic or the prompt is unavailable.
    """
    if not user_query:
        return user_query
    if _anthropic_client is None:
        return user_query

    prompt_text = _load_prompt_text(base_prompt_filename)
    if not prompt_text:
        return user_query

    try:
        model = os.getenv(model_env_var, os.getenv("LLM_MODEL_ROUTER", "claude-haiku-4-5-20251001"))
        instruction = (
            f"{prompt_text}\n\n"
            "Task: Given the user's input below, produce a concise search query suitable for vector search.\n"
            "Return ONLY the query string, with no quotes and no extra text.\n\n"
            f"User input:\n{user_query}\n\nOutput:"
        )
        msg = _anthropic_client.messages.create(
            model=model,
            max_tokens=5000,
            temperature=0.0,
            system=(
                "You transform inputs into optimized, compact search queries. "
                "Your output must contain ONLY the final query text."
            ),
            messages=[{"role": "user", "content": instruction}],
        )
        text_blocks = [getattr(b, "text", "") for b in msg.content if getattr(b, "type", None) == "text"]
        out = ("\n".join(text_blocks)).strip().strip("\"'")
        return out or user_query
    except Exception:
        return user_query

def _matches_to_output(matches: List) -> List[Dict]:
    out = []
    for m in matches or []:
        if isinstance(m, dict):
            pid = m.get("id")
            score = m.get("score")
            meta = m.get("metadata")
        else:
            pid = getattr(m, "id", None)
            score = getattr(m, "score", None)
            meta = getattr(m, "metadata", None)
        out.append({"product_id": pid, "score": score, "metadata": meta})
    return out

def _extract_sku(meta: Dict) -> Optional[str]:
    if not isinstance(meta, dict):
        return None
    candidates = [
        "sku", "SKU", "product_sku", "sku_id", "Sku", "SkuId", "productSku",
        "Product SKU", "product_sku_id"
    ]
    for k in candidates:
        if k in meta and isinstance(meta[k], (str, int)):
            return str(meta[k]).strip()
    # Sometimes SKU embedded in id-like fields
    for k, v in meta.items():
        if isinstance(v, str) and k.lower() in ("id", "pid", "productid", "product_id") and len(v) < 64:
            # Fallback only if it looks like human SKU (no spaces)
            if " " not in v:
                return v.strip()
    return None

def embed_text(text: str) -> List[float]:
    """Embed text using OpenAI's text-embedding-3-large model.

    Requires OPENAI_API_KEY in environment. Returns a list[float] suitable for Pinecone.
    """
    if not text:
        return []
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set; cannot call text-embedding-3-large")

    model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
    client = OpenAI(api_key=api_key)
    
    try:
        resp = client.embeddings.create(model=model, input=text)
        vec = [float(v) for v in resp.data[0].embedding]
        
        # Optional dimension validation
        if _pc_expected_dim is not None and len(vec) != _pc_expected_dim:
            raise RuntimeError(
                f"Embedding dimension {len(vec)} != PINECONE_DIMENSION={_pc_expected_dim}. "
                "Make sure your Pinecone index dimension matches the embedding model."
            )
        return vec
    except Exception as e:
        raise RuntimeError(f"OpenAI Embedding API request failed: {e}")

@beta_tool(
    name="resolve_product",
    description=(
        "Resolves a product by its exact name from Pinecone. "
        "Returns candidate products with ids and metadata so downstream tools can operate by id."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Exact product name to search for"},
            "top_k": {"type": "integer", "minimum": 10, "description": "Number of candidates to return"}
        },
        "required": ["name", "top_k"],
        "additionalProperties": False
    }
)

def resolve_product(name: str, top_k: int, category: str = None) -> str:
    query_text = name
    vec = embed_text(query_text)
    
    # Map input category to the exact Pinecone category structure
    category_mapping = {
        'lip_balm_treatment': {
            'category': 'Makeup',
            'sub_category': 'Lip',
            'leaf_level_category': 'Lip Balm & Treatment'
        },
        'lipstick': {
            'category': 'Makeup',
            'sub_category': 'Lip',
            'leaf_level_category': 'Lipstick'
        },
        'lip_liner': {
            'category': 'Makeup',
            'sub_category': 'Lip',
            'leaf_level_category': 'Lip Liner'
        }
    }
    
    # Prepare filter based on the exact category structure
    filter_condition = None
    if category and category.lower() != 'all':
        if category.lower() in category_mapping:
            cat_info = category_mapping[category.lower()]
            filter_condition = {
                "$and": [
                    {"category": {"$eq": cat_info['category']}},
                    {"sub_category": {"$eq": cat_info['sub_category']}},
                    {"leaf_level_category": {"$eq": cat_info['leaf_level_category']}}
                ]
            }
    
    # Build query parameters
    query_params = {
        "vector": vec,
        "top_k": top_k,
        "include_values": False,
        "include_metadata": True,
        "namespace": _pc_namespace
    }
    
    # Add filter if category is specified
    if filter_condition:
        query_params["filter"] = filter_condition
    
    try:
        results = index.query(**query_params)
        matches = getattr(results, "matches", []) or results.get("matches", [])
        candidates = _matches_to_output(matches)
    except Exception as e:
        return json.dumps({"error": f"pinecone query failed: {e}"})

    # If no candidates, return early
    if not candidates:
        return json.dumps({"content": [], "selected_sku": None, "selected_product": None})

    # Prepare a compact list for LLM selection
    llm_items = []
    for c in candidates:
        meta = c.get("metadata") or {}
        item = {
            "product_id": c.get("product_id"),
            "name": meta.get("name") or meta.get("product_name") or meta.get("title"),
            "brand": meta.get("brand"),
            "category": meta.get("leaf_level_category") or meta.get("category"),
            "sku": _extract_sku(meta),
        }
        llm_items.append(item)

    selected_sku: Optional[str] = None
    # Ask LLM to pick the best SKU if possible
    if _anthropic_client is not None:
        try:
            instruction = (
                "You will be given a user product query and a list of candidate products. "
                "Pick the single best matching item's SKU. Return ONLY the SKU text. "
                "If a SKU is missing for the best match, return the product_id instead.\n\n"
                f"User query: {name}\n\nCandidates (JSON):\n{json.dumps(llm_items, ensure_ascii=False)}\n\nOutput:"
            )
            msg = _anthropic_client.messages.create(
                model=os.getenv("LLM_MODEL_RESOLVE", "claude-haiku-4-5-20251001"),
                max_tokens=15000,
                temperature=0.0,
                system=(
                    "Return ONLY the SKU or product_id with no extra text, quotes, or formatting. "
                    "If multiple look similar, choose the one with the closest name match to the query."
                ),
                messages=[{"role": "user", "content": instruction}],
            )
            text_blocks = [getattr(b, "text", "") for b in msg.content if getattr(b, "type", None) == "text"]
            choice = ("\n".join(text_blocks)).strip()
            # Strip quotes if any
            choice = choice.strip('"\'')
            if choice:
                selected_sku = choice
        except Exception:
            selected_sku = None

    # If no LLM or no choice, try the first candidate with a SKU
    if not selected_sku:
        for it in llm_items:
            if it.get("sku"):
                selected_sku = it["sku"]
                break
        # Fallback to first product_id
        if not selected_sku and llm_items:
            selected_sku = llm_items[0].get("product_id")

    # Re-query Pinecone filtered by the chosen SKU (or matching id) to get the exact product metadata

    selected_products = []
    try:
        if selected_sku:
            # Build a permissive filter that checks common sku key names
            sku_filters = [
                {"sku": {"$eq": selected_sku}},
                {"SKU": {"$eq": selected_sku}},
                {"product_sku": {"$eq": selected_sku}},
                {"sku_id": {"$eq": selected_sku}},
                {"product_id": {"$eq": selected_sku}},
            ]
            requery_params = {
                "vector": vec,  # reuse query vector; filter will narrow to exact doc
                "top_k": 10,
                "include_values": False,
                "include_metadata": True,
                "namespace": _pc_namespace,
                "filter": {"$or": sku_filters},
            }
            re_res = index.query(**requery_params)
            re_matches = getattr(re_res, "matches", []) or re_res.get("matches", [])
            if re_matches:
                # Convert all matches
                selected_products = _matches_to_output(re_matches)
    except Exception:
        selected_products = []

    return json.dumps({
        "selected_sku": selected_sku,
        "selected_products": selected_products
    })

@beta_tool(
    name="retrieve_similar_products",
    description=(
        "Retrieves a list of products that are similar to the given product description in the lip-care domain "
        "(lipsticks, lip balm & treatments). Input: product description and top_k results."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "product_name": {
                "type": "string",
                "description": "Complete product name to anchor similarity (e.g., brand + line + shade)"
            },
            "top_k": {
                "type": "integer",
                "minimum": 10,
                "description": "Number of similar products to return"
            },
            "selected_sku": {
                "type": "string",
                "description": "Optional SKU resolved from a prior resolve step"
            },
            "selected_products": {
                "type": "array",
                "description": "Optional prior resolved products array to improve similarity selection"
            }
        },
        "required": ["top_k", "product_name"],
        "additionalProperties": False
    }
)
def retrieve_similar_products(
    top_k: int,
    product_name: str,
    category: str = None,
    selected_sku: Optional[str] = None,
    selected_products: Optional[List[Dict]] = None,
) -> str:
    """
    Find similar products based on the provided complete product name (used as the reference anchor).
    
    Args:
        top_k: Number of similar products to return (minimum 10)
        product_name: Complete product name to anchor similarity (brand + line + shade)
        category: Optional category to filter by (lipstick, lip_balm_treatment, or lip_liner)
    """
    # Map input category to the exact Pinecone category structure
    category_mapping = {
        'lip_balm_treatment': {
            'category': 'Makeup',
            'sub_category': 'Lip',
            'leaf_level_category': 'Lip Balm & Treatment'
        },
        'lipstick': {
            'category': 'Makeup',
            'sub_category': 'Lip',
            'leaf_level_category': 'Lipstick'
        },
        'lip_liner': {
            'category': 'Makeup',
            'sub_category': 'Lip',
            'leaf_level_category': 'Lip Liner'
        }
    }
    
    # Prepare filter based on the exact category structure
    filter_condition = None
    if category and category.lower() != 'all':
        if category.lower() in category_mapping:
            cat_info = category_mapping[category.lower()]
            filter_condition = {
                "$and": [
                    {"category": {"$eq": cat_info['category']}},
                    {"sub_category": {"$eq": cat_info['sub_category']}},
                    {"leaf_level_category": {"$eq": cat_info['leaf_level_category']}}
                ]
            }
    
    # Optionally refine the description into a compact search query using a dedicated prompt/LLM.
    # If we have prior resolved context (selected_products), include it to improve query specificity.
    first_layer_input = {
        "reference_description": product_name,
        "selected_sku": selected_sku,
        "selected_products": (selected_products or []),  # keep prompt small
    }
    augmented_query = (
        f"REFERENCE:\n{product_name}\n\n"
        f"CONTEXT_SELECTED (JSON):\n{json.dumps(first_layer_input, ensure_ascii=False)}"
    )
    refined_desc = _refine_query_with_llm("similar_products_prompt.txt", augmented_query, "LLM_MODEL_SIMILAR")
    vec = embed_text(refined_desc)  # embed model call
    
    # Build query parameters
    query_params = {
        "vector": vec,
        "top_k": top_k,
        "include_values": False,
        "include_metadata": True,
        "namespace": _pc_namespace
    }
    
    # Add filter if category is specified
    if filter_condition:
        query_params["filter"] = filter_condition
    
    # Query Pinecone with optional category filter
    try:
        results = index.query(**query_params)
    except Exception as e:
        return json.dumps({"error": f"pinecone query failed: {e}"})
    matches = getattr(results, "matches", []) or results.get("matches", [])
    output = _matches_to_output(matches)

    # Post-retrieval LLM step: use the dedicated prompt to select/rank similar products.
    # Return plain text directly (no JSON parsing). Strip code fences if present.
    prompt_text = _load_prompt_text("similar_products_prompt.txt")
    if _anthropic_client is not None and prompt_text:
        try:
            model = os.getenv("LLM_MODEL_SIMILAR", os.getenv("LLM_MODEL_ROUTER", "claude-haiku-4-5-20251001"))
            input_payload = {
                "reference_description": product_name,
                "target_category": category,
                "retrieved_items": output,
                "top_k": top_k,
            }
            instruction = (
                f"{prompt_text}\n\n"
                "Use the inputs below to produce the final output.\n"
                "Inputs (JSON):\n"
                f"{json.dumps(input_payload, ensure_ascii=False)}\n\n"
                "Return only the final output, no code fences."
            )
            msg = _anthropic_client.messages.create(
                model=model,
                max_tokens=10000,
                temperature=0.0,
                messages=[{"role": "user", "content": instruction}],
            )
            text_blocks = [getattr(b, "text", "") for b in msg.content if getattr(b, "type", None) == "text"]
            llm_out = ("\n".join(text_blocks)).strip()
            if llm_out.startswith("```"):
                llm_out = llm_out.strip().lstrip("`")
                llm_out = "\n".join(llm_out.splitlines()[1:]) if "\n" in llm_out else llm_out
                if llm_out.endswith("```"):
                    llm_out = llm_out[:-3].strip()
            return llm_out
        except Exception:
            pass

    # Fallback: return raw retrieved items
    return json.dumps({"content": output})

@beta_tool(
    name="retrieve_use_with_products",
    description=(
        "Retrieves complementary products based on a search query and category. "
        "The query should describe the type of complementary products to find. "
        "For example: 'lip balm that pairs well with matte lipstick' or 'lip liner that matches red lipstick'"
    ),
    input_schema={
        "type": "object",
        "properties": {
            "product_name": {
                "type": "string", 
                "description": "Complete product name for which to find complementary items"
            },
            "category": {
                "type": "string",
                "enum": ["lipstick", "Lip Balm & Treatment", "Lip Liner"],
                "description": "Category to search within (lipstick or Lip Balm & Treatment or Lip Liner)"
            },
            "top_k": {
                "type": "integer", 
                "minimum": 5, 
                "description": "Number of complementary products to return"
            },
            "selected_sku": {
                "type": "string",
                "description": "Optional SKU resolved from a prior resolve step"
            },
            "selected_products": {
                "type": "array",
                "description": "Optional prior resolved products array to improve complement selection"
            }
        },
        "required": ["product_name", "category", "top_k"],
        "additionalProperties": False
    }
)
def retrieve_use_with_products(product_name: str,
                             category: str = None,
                             top_k: int = 5,
                             selected_sku: Optional[str] = None,
                             selected_products: Optional[List[Dict]] = None) -> str:
    try:
        # Prepare context payload with prior resolved context when available
        first_layer_input = {
            "user_query": product_name,
            "selected_sku": selected_sku,
            "selected_products": (selected_products or []),  # keep prompt compact
        }
        context_payload = (
            f"PRODUCT NAME: {product_name}\n"
            f"CONTEXT_SELECTED (JSON): {json.dumps(first_layer_input, ensure_ascii=False)}"
        )

        # Map input category to the exact Pinecone category structure
        category_mapping = {
            'lip_balm_treatment': {
                'category': 'Makeup',
                'sub_category': 'Lip',
                'leaf_level_category': 'Lip Balm & Treatment'
            },
            'lipstick': {
                'category': 'Makeup',
                'sub_category': 'Lip',
                'leaf_level_category': 'Lipstick'
            },
            'lip_liner': {
                'category': 'Makeup',
                'sub_category': 'Lip',
                'leaf_level_category': 'Lip Liner'
            }
        }

        # Generate a single batch that includes categories and their queries, then parse
        per_category_results: Dict[str, Any] = {}
        pre_prompt_text = _load_prompt_text("use_it_with_pre_prompt.txt")

        parsed_queries: Dict[str, List[str]] = {}
        if _anthropic_client is not None and pre_prompt_text:
            try:
                model = os.getenv("LLM_MODEL_USE_WITH", os.getenv("LLM_MODEL_ROUTER", "claude-haiku-4-5-20251001"))
                pre_instruction = (
                    f"{pre_prompt_text}\n\n"
                    f"{context_payload}\n"
                    f"INPUT CATEGORY: {category or ''}\n\n"
                    "batch 3\n"
                    "Group queries under [CATEGORY: ...] headers."
                )
                msg = _anthropic_client.messages.create(
                    model=model,
                    max_tokens=400,
                    temperature=0.3,
                    messages=[{"role": "user", "content": pre_instruction}],
                )
                text_blocks = [getattr(b, "text", "") for b in msg.content if getattr(b, "type", None) == "text"]
                raw = ("\n".join(text_blocks)).strip()
                lines = [l.strip() for l in raw.splitlines() if l.strip()]
                current_cat = None
                for l in lines:
                    if l.lower().startswith("[category:"):
                        # Strict header: one of lipstick, lip_liner, lip_balm_treatment
                        header = l[10:-1].strip().lower()  # content between [CATEGORY: and ]
                        if header == 'lip_liner':
                            current_cat = 'lip_liner'
                        elif header == 'lip_balm_treatment':
                            current_cat = 'lip_balm_treatment'
                        elif header == 'lipstick':
                            current_cat = 'lipstick'
                        else:
                            current_cat = None
                        if current_cat and current_cat not in parsed_queries:
                            parsed_queries[current_cat] = []
                        continue
                    if current_cat and not l.startswith('['):
                        parsed_queries[current_cat].append(l)
                # Trim to 3 per category
                for k in list(parsed_queries.keys()):
                    parsed_queries[k] = parsed_queries[k][:3]
            except Exception:
                parsed_queries = {}

        # Do not filter parsed categories by user-provided category. Run all LLM-decided categories.

        # Fallback: if no categories/queries were parsed, attempt a single-category refinement as backup
        if not parsed_queries:
            fallback_cat = (category.lower() if category and category.lower() in category_mapping else 'lip_balm_treatment')
            parsed_queries = {fallback_cat: []}
            if pre_prompt_text:
                single_q_instruction = (
                    f"{pre_prompt_text}\n\n"
                    f"{context_payload}\n"
                    f"INPUT CATEGORY: {fallback_cat}\n\n"
                    "Generate ONE query. Output only the query text."
                )
                try:
                    model = os.getenv("LLM_MODEL_USE_WITH", os.getenv("LLM_MODEL_ROUTER", "claude-haiku-4-5-20251001"))
                    msg = _anthropic_client.messages.create(
                        model=model,
                        max_tokens=60,
                        temperature=0.2,
                        messages=[{"role": "user", "content": single_q_instruction}],
                    )
                    text_blocks = [getattr(b, "text", "") for b in msg.content if getattr(b, "type", None) == "text"]
                    one_q = ("\n".join(text_blocks)).strip()
                    if one_q:
                        parsed_queries[fallback_cat] = [one_q]
                except Exception:
                    pass

        # Execute Pinecone search per parsed category and query, then aggregate raw results
        for cat_key, queries in parsed_queries.items():
            cat_info = category_mapping.get(cat_key)
            if not cat_info:
                continue
            cat_results: Dict[str, Any] = {}
            for q in queries:
                vec = embed_text(q)
                if not vec:
                    cat_results[q] = {"error": "Failed to generate embedding for query"}
                    continue
                filter_condition = {
                    "$and": [
                        {"category": {"$eq": cat_info['category']}},
                        {"sub_category": {"$eq": cat_info['sub_category']}},
                        {"leaf_level_category": {"$eq": cat_info['leaf_level_category']}}
                    ]
                }
                query_params = {
                    "vector": vec,
                    "top_k": top_k,
                    "include_values": False,
                    "include_metadata": True,
                    "namespace": _pc_namespace,
                    "filter": filter_condition,
                }
                try:
                    results = index.query(**query_params)
                    matches = getattr(results, "matches", []) or results.get("matches", [])
                    output = _matches_to_output(matches)
                except Exception as e:
                    cat_results[q] = {"error": f"pinecone query failed: {e}"}
                    continue

                # Directly return raw retrieved items without per-query post-ranking
                cat_results[q] = {"content": output}

            per_category_results[cat_key] = {"queries": cat_results}

        # Final packaging step: use use_it_with_prompt.txt and return plain text (no JSON parsing here)
        try:
            rank_prompt_text = _load_prompt_text("use_it_with_prompt.txt")
            if _anthropic_client is not None and rank_prompt_text:
                model = os.getenv("LLM_MODEL_USE_WITH", os.getenv("LLM_MODEL_ROUTER", "claude-haiku-4-5-20251001"))
                input_payload = {
                    "product_name": product_name,
                    "by_category": per_category_results,
                    "top_k": top_k,
                    "selected_sku": selected_sku,
                    "selected_products": (selected_products or []),
                }
                instruction = (
                    f"{rank_prompt_text}\n\n"
                    "Use the inputs below to produce the final JSON output.\n"
                    "Inputs (JSON):\n"
                    f"{json.dumps(input_payload, ensure_ascii=False)}\n\n"
                    "Return only the final output, no code fences."
                )
                msg = _anthropic_client.messages.create(
                    model=model,
                    max_tokens=20000,
                    temperature=0.0,
                    messages=[{"role": "user", "content": instruction}],
                )
                text_blocks = [getattr(b, "text", "") for b in msg.content if getattr(b, "type", None) == "text"]
                llm_out = ("\n".join(text_blocks)).strip()
                # Strip code fences if present
                if llm_out.startswith("```"):
                    llm_out = llm_out.strip().lstrip("`")
                    llm_out = "\n".join(llm_out.splitlines()[1:]) if "\n" in llm_out else llm_out
                    if llm_out.endswith("```"):
                        llm_out = llm_out[:-3].strip()
                return llm_out
        except Exception:
            pass

        # Fallback: return raw by_category JSON if packaging fails
        return json.dumps({"by_category": per_category_results})
        
    except Exception as e:
        return json.dumps({"error": f"Search failed: {str(e)}"})

@beta_tool(
    name="general_product_qna",
    description=(
        "Answers general product questions directly from Pinecone results without calling resolve_product. "
        "Use for simple QnA like ingredients, finishes, brand details, or quick facts."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "User's natural-language question (e.g., 'What ingredient use in Dr. PawPaw Lip & Eye Balm?')"},
            "category": {"type": "string", "enum": ["lipstick", "lip_balm_treatment", "lip_liner"], "description": "Optional category to filter search"},
            "top_k": {"type": "integer", "minimum": 5, "description": "How many candidates to fetch from Pinecone before answering"}
        },
        "required": ["query"],
        "additionalProperties": False
    }
)
def general_product_qna(query: str,
                        category: Optional[str] = None,
                        top_k: int = 5) -> str:
    """
    General QnA over product corpus. Does NOT call resolve_product.
    Flow: embed query -> Pinecone -> prompt1.txt to compose the answer -> return plain text.
    """
    try:
        # Optional category filter aligned to Pinecone schema
        category_mapping = {
            'lip_balm_treatment': {
                'category': 'Makeup',
                'sub_category': 'Lip',
                'leaf_level_category': 'Lip Balm & Treatment'
            },
            'lipstick': {
                'category': 'Makeup',
                'sub_category': 'Lip',
                'leaf_level_category': 'Lipstick'
            },
            'lip_liner': {
                'category': 'Makeup',
                'sub_category': 'Lip',
                'leaf_level_category': 'Lip Liner'
            }
        }

        # Embed the raw user query (no resolve step)
        vec = embed_text(query)
        if not vec:
            return "Sorry, I couldn't process your question right now. Please try again."

        query_params = {
            "vector": vec,
            "top_k": top_k,
            "include_values": False,
            "include_metadata": True,
            "namespace": _pc_namespace
        }
        if category and category.lower() in category_mapping:
            cat_info = category_mapping[category.lower()]
            query_params["filter"] = {
                "$and": [
                    {"category": {"$eq": cat_info['category']}},
                    {"sub_category": {"$eq": cat_info['sub_category']}},
                    {"leaf_level_category": {"$eq": cat_info['leaf_level_category']}}
                ]
            }

        try:
            results = index.query(**query_params)
            matches = getattr(results, "matches", []) or results.get("matches", [])
            retrieved = _matches_to_output(matches)
        except Exception as e:
            return f"Unable to retrieve data right now: {e}"

        # Compose final answer using prompt1.txt
        prompt_text = _load_prompt_text("prompt1.txt")
        if _anthropic_client is not None and prompt_text:
            try:
                model = os.getenv("LLM_MODEL_QNA", os.getenv("LLM_MODEL_ROUTER", "claude-haiku-4-5-20251001"))
                input_payload = {
                    "query": query,
                    "category": category,
                    "retrieved_items": retrieved,
                }
                instruction = (
                    f"{prompt_text}\n\n"
                    "Use the inputs below to answer the user's question using only the retrieved items when possible.\n"
                    "If asking about ingredients or product facts, extract from metadata fields.\n"
                    "Inputs (JSON):\n"
                    f"{json.dumps(input_payload, ensure_ascii=False)}\n\n"
                    "Return only the final answer (no code fences)."
                )
                msg = _anthropic_client.messages.create(
                    model=model,
                    max_tokens=2000,
                    temperature=0.2,
                    messages=[{"role": "user", "content": instruction}],
                )
                text_blocks = [getattr(b, "text", "") for b in msg.content if getattr(b, "type", None) == "text"]
                llm_out = ("\n".join(text_blocks)).strip()
                if llm_out.startswith("```"):
                    llm_out = llm_out.strip().lstrip("`")
                    llm_out = "\n".join(llm_out.splitlines()[1:]) if "\n" in llm_out else llm_out
                    if llm_out.endswith("```"):
                        llm_out = llm_out[:-3].strip()
                return llm_out
            except Exception:
                pass

        # Fallback: minimal textual summary from top match
        try:
            top = retrieved[0] if retrieved else None
            if top and isinstance(top, dict):
                meta = top.get("metadata", {}) or {}
                name = meta.get("product_name") or meta.get("title") or meta.get("name") or "the product"
                return f"Here's what I found about {name}: {json.dumps(meta, ensure_ascii=False)[:800]}..."
        except Exception:
            pass
        return "I couldn't find a confident answer right now. Please try rephrasing your question."
    except Exception as e:
        return f"Error: {e}"
