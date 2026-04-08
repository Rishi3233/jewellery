
import os
import io
import time
import base64
import requests
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

import anthropic
from PIL import Image

from .config import INSPIRATION_OUTPUT_DIR
from .search import search_similar


# -----------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class CatalogItem:
    title: str
    description: str
    reference_image_path: str
    reference_product_id: str
    generated_image_path: Optional[str] = None


# -----------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def image_to_base64(image: Image.Image) -> str:
    """Resize to max 1024px and convert to base64 JPEG."""
    img_copy = image.copy()
    img_copy.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    img_copy.convert("RGB").save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# -----------------------------------------------------------------------
# Stage 1 — Claude Vision: Validate + Describe
# ---------------------------------------------------------------------------

def extract_style_tags(image: Image.Image, style_notes: str = "") -> List[str]:
    """
    Validates image contains jewellery and describes it.

    Returns:
      ["REJECTED: reason"]   — not jewellery
      ["description"]        — jewellery found
      ["Error: message"]     — API failure
    """
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        return ["Error: ANTHROPIC_API_KEY missing from .env file."]

    client = anthropic.Anthropic(api_key=api_key)
    img_b64 = image_to_base64(image)

    prompt = (
        "You are a master jewelry appraiser. First, determine if the image is actually jewelry. "
        "If it is NOT jewelry (e.g., an anime character, a landscape, a random object), "
        "you MUST start your response with the exact word 'REJECTED:' followed by what it actually is. "
        "If it IS jewelry, describe this piece in one highly detailed, professional sentence "
        "focusing on materials, stones, and geometry. Return ONLY plain text. "
        "Do not use any markdown formatting, bolding, or headers."
    )
    if style_notes.strip():
        prompt += (
            f" CRITICAL INSTRUCTION: Alter your description to strictly incorporate "
            f"these user changes: {style_notes}"
        )

    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=150,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": img_b64,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )

        print(f"\n--- API USAGE (extract_style_tags) ---")
        print(f"Input Tokens : {response.usage.input_tokens}")
        print(f"Output Tokens: {response.usage.output_tokens}")
        print(f"--------------------------------------\n")

        return [response.content[0].text.strip()]

    except Exception as e:
        print(f"[inspiration] extract_style_tags error: {e}")
        return [f"Error: {str(e)}"]


# -----------------------------------------------------------------------
# Stage 2 — Novita AI: Generate concept image
# ---------------------------------------------------------------------------

def run_novita_generation(
    image: Image.Image,
    prompt: str,
    mode: str,
    negative_notes: str = "",
) -> Optional[str]:
    """
    Generate concept image using Novita AI.
    Saves to data/outputs/inspiration/ folder.

    Returns: path to saved image or None on failure.
    """
    api_key = os.getenv("NOVITA_API_KEY", "")
    if not api_key:
        print("[inspiration] NOVITA_API_KEY missing — skipping image generation.")
        return None

    # FIX: Create inspiration output folder if it doesn't exist
    INSPIRATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    img_b64 = image_to_base64(image)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    if "Sketch" in mode:
        url = "https://api.novita.ai/v3/async/txt2img"
        payload = {
            "extra": {"response_image_type": "jpeg"},
            "request": {
                "model_name": "sd_xl_base_1.0.safetensors",
                "prompt": f"photorealistic 8k commercial jewelry photography, {prompt}",
                "negative_prompt": f"illustration, cartoon, deformed, blurry, {negative_notes}",
                "width": 1024, "height": 1024,
                "image_num": 1, "steps": 30,
                "guidance_scale": 7.5, "sampler_name": "Euler a",
                "controlnet": {
                    "units": [{
                        "model_name": "control_v11p_sd15_lineart",
                        "image_base64": img_b64,
                        "strength": 1.0,
                    }]
                },
            },
        }
    else:
        url = "https://api.novita.ai/v3/async/img2img"
        payload = {
            "extra": {"response_image_type": "jpeg"},
            "request": {
                "model_name": "sd_xl_base_1.0.safetensors",
                "image_base64": img_b64,
                "prompt": f"highly detailed jewelry photography, {prompt}",
                "negative_prompt": f"blurry, distorted, {negative_notes}",
                "strength": 0.85,
                "width": 1024, "height": 1024,
                "image_num": 1, "steps": 30,
                "guidance_scale": 7.5, "sampler_name": "Euler a",
            },
        }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30).json()
        task_id = response.get("task_id")

        if not task_id:
            print(f"[inspiration] Novita task creation failed: {response}")
            return None

        print(f"[inspiration] Novita task submitted: {task_id}")

        for attempt in range(20):
            time.sleep(2)
            poll_url = f"https://api.novita.ai/v3/async/task-result?task_id={task_id}"
            poll_res = requests.get(poll_url, headers=headers, timeout=30).json()
            status = poll_res.get("task", {}).get("status", "")
            print(f"[inspiration] Novita poll {attempt + 1}: {status}")

            if status == "TASK_STATUS_SUCCEED":
                images = poll_res.get("images", [])
                if images:
                    img_url = images[0].get("image_url")
                    img_data = requests.get(img_url, timeout=60).content

                    # FIX: Save to data/outputs/inspiration/ instead of data/outputs/
                    save_path = INSPIRATION_OUTPUT_DIR / f"novita_render_{task_id[:8]}.jpg"
                    with open(save_path, "wb") as f:
                        f.write(img_data)

                    print(f"[inspiration] Novita image saved: {save_path}")
                    return str(save_path)

            elif status in ("TASK_STATUS_FAILED", "TASK_STATUS_CANCELLED"):
                print(f"[inspiration] Novita task failed: {poll_res}")
                return None

    except Exception as e:
        print(f"[inspiration] Novita API error: {e}")
        return None

    print("[inspiration] Novita timed out.")
    return None


# ---------------------------------------------------------------------------
# Stage 3 — FAISS Catalog Search
# ---------------------------------------------------------------------------

def generate_concepts_from_image(
    image: Image.Image,
    claude_analysis: str = "",
) -> List[CatalogItem]:
    """
    Run CLIP + FAISS search and return closest catalog matches.
    """
    try:
        faiss_results = search_similar(image, top_k=3)

        if not faiss_results:
            print("[inspiration] No FAISS results found.")
            return []

        matches = []
        for res in faiss_results:
            category = res.metadata.get("category", "Jewellery Match").title()
            price = res.metadata.get("price", "N/A")
            matches.append(
                CatalogItem(
                    title=f"{category} Design",
                    description=f"Match Quality: {res.score * 100:.1f}% | Est. Price: ₹{price}",
                    reference_image_path=str(res.image_path),
                    reference_product_id=str(res.product_id),
                    generated_image_path=None,
                )
            )
        return matches

    except Exception as e:
        print(f"[inspiration] FAISS search error: {e}")
        return []