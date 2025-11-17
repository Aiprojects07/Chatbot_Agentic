# tools/product_tools.py

import os
import json
from typing import Optional, List, Dict
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
            "top_k": {"type": "integer", "minimum": 8, "description": "Number of candidates to return"}
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
        output = _matches_to_output(matches)
        return json.dumps({"content": output})
    except Exception as e:
        return json.dumps({"error": f"pinecone query failed: {e}"})

@beta_tool(
    name="retrieve_similar_products",
    description=(
        "Retrieves a list of products that are similar to the given product description in the lip-care domain "
        "(lipsticks, lip balm & treatments). Input: product description and top_k results."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "product_description": {
                "type": "string",
                "description": "Description of the product to find similar items for"
            },
            "top_k": {
                "type": "integer",
                "minimum": 10,
                "description": "Number of similar products to return"
            }
        },
        "required": ["top_k", "product_description"],
        "additionalProperties": False
    }
)
def retrieve_similar_products(
    top_k: int,
    product_description: str,
    category: str = None
) -> str:
    """
    Find similar products based on the provided product description.
    
    Args:
        top_k: Number of similar products to return (minimum 10)
        product_description: Detailed description of the reference product
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
    
    vec = embed_text(product_description)  # embed model call
    
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
            "query": {
                "type": "string", 
                "description": "Search query describing the complementary products to find"
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
            }
        },
        "required": ["query", "category", "top_k"],
        "additionalProperties": False
    }
)
def retrieve_use_with_products(query: str,
                             category: str = None,
                             top_k: int = 5) -> str:
    try:
        # Generate embedding for the search query
        vec = embed_text(query)
        
        if vec is None:
            return json.dumps({"error": "Failed to generate embedding for the query"})
            
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
        
        # Query Pinecone with the generated vector and optional category filter
        query_params = {
            "vector": vec,
            "top_k": top_k,
            "include_values": False,
            "include_metadata": True,
            "namespace": _pc_namespace
        }
        
        if filter_condition:
            query_params["filter"] = filter_condition
            
        results = index.query(**query_params)
        
        matches = getattr(results, "matches", []) or results.get("matches", [])
        output = _matches_to_output(matches)
        return json.dumps({"content": output})
        
    except Exception as e:
        return json.dumps({"error": f"Search failed: {str(e)}"})
