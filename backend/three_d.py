
import base64
import io
import json
import os
import re
import textwrap
from pathlib import Path
from typing import Optional, Tuple

import requests
from PIL import Image

from .config import THREE_D_OUTPUT_DIR


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _image_to_base64(image: Image.Image) -> str:
    """Convert PIL image to base64 JPEG string."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def _call_claude_vision(image: Image.Image, prompt: str, max_tokens: int = 2048) -> str:
    """Send image + prompt to Claude Vision API and return text response."""
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is not set. "
            "Add it to your .env file: ANTHROPIC_API_KEY=sk-ant-..."
        )

    image_b64 = _image_to_base64(image)

    payload = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_b64,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }

    response = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json=payload,
        timeout=60,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Claude API error {response.status_code}: {response.text}"
        )

    data = response.json()
    return data["content"][0]["text"]


def _parse_json_safe(text: str) -> dict:
    """Extract JSON from Claude response, handling markdown code fences."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

ANALYSIS_PROMPT = """
You are an expert jewellery CAD designer with 20 years of experience.

Analyze this jewellery image carefully and return ONLY a JSON object (no markdown, no extra text) with this exact structure:

{
  "jewellery_type": "Ring / Necklace / Earring / Bangle / Pendant / Nosepin",
  "overall_shape": "Description of the overall 3D shape",
  "dimensions": {
    "estimated_width_mm": 18,
    "estimated_height_mm": 8,
    "estimated_depth_mm": 4,
    "band_thickness_mm": 2
  },
  "components": [
    {
      "name": "component name (e.g. Main Band, Center Stone, Side Prongs)",
      "shape": "geometric shape description",
      "material": "metal or stone type",
      "finish": "polished / matte / antique / textured",
      "position": "where it sits in the design"
    }
  ],
  "metal": "Yellow Gold / White Gold / Rose Gold / Silver / Platinum",
  "karat": "22K / 18K / 14K / 925 Silver / N/A",
  "stones": [
    {
      "type": "Diamond / Ruby / Emerald / Pearl / None",
      "cut": "Round / Oval / Cushion / Marquise / Pear / N/A",
      "setting": "Prong / Bezel / Pave / Channel / Halo / N/A",
      "count": 1,
      "estimated_size_mm": 5
    }
  ],
  "design_style": "Traditional / Contemporary / Vintage / Fusion / Minimalist",
  "symmetry": "Symmetric / Asymmetric / Radial",
  "cad_notes": "Important notes for the CAD designer about tricky parts, undercuts, stone seats, prong placement etc.",
  "complexity_level": "Simple / Medium / Complex",
  "estimated_cad_hours_without_ai": 6,
  "estimated_cad_hours_with_ai_reference": 1.5
}

Be as accurate and detailed as possible. All measurements are estimates based on typical jewellery proportions.
"""

SVG_PROMPT = """
You are an expert jewellery technical illustrator.

Based on this jewellery image, create a technical wireframe SVG showing THREE views:
1. Front View (left panel)
2. Side View (center panel)
3. Top View (right panel)

Return ONLY the raw SVG code starting with <svg and ending with </svg>.
No markdown, no explanation, no code fences.

Requirements:
- SVG viewBox="0 0 900 400"
- White background rectangle covering full SVG
- Three equal panels separated by vertical dashed lines
- Each panel has a label: "FRONT VIEW", "SIDE VIEW", "TOP VIEW"
- Draw clean geometric wireframe shapes using lines, circles, ellipses, rectangles, and paths
- Use stroke="#1a1a1a" fill="none" strokeWidth="1.5" for main outlines
- Use stroke="#888888" strokeDasharray="4,3" for hidden/internal lines
- Use stroke="#cc8800" fill="none" for stone positions
- Add dimension lines with arrows showing width and height
- Add small text labels for key components (Band, Stone, Setting, etc.)
- Style: clean technical drawing, no fills except white background
- Make it look like a professional jewellery CAD technical drawing
"""


# ---------------------------------------------------------------------------
# SVG & file generators
# ---------------------------------------------------------------------------

def _generate_fallback_svg(jewellery_type: str, spec: dict) -> str:
    """Fallback SVG if Claude SVG generation fails."""
    jtype = jewellery_type.lower()
    dim_color = "#0055cc"
    stone_color = "#cc8800"
    hidden = "#aaaaaa"
    label_color = "#333333"

    metal = spec.get("metal", "Gold") if spec else "Gold"
    karat = spec.get("karat", "") if spec else ""
    style = spec.get("design_style", "") if spec else ""
    dims = spec.get("dimensions", {}) if spec else {}
    width_mm = dims.get("estimated_width_mm", 18)
    height_mm = dims.get("estimated_height_mm", 8)
    complexity = spec.get("complexity_level", "Medium") if spec else "Medium"

    if "ring" in jtype:
        front = """
        <ellipse cx="150" cy="200" rx="55" ry="55" stroke="#1a1a1a" fill="none" stroke-width="2"/>
        <ellipse cx="150" cy="200" rx="38" ry="38" stroke="#1a1a1a" fill="none" stroke-width="1.5"/>
        <ellipse cx="150" cy="155" rx="18" ry="12" stroke="#cc8800" fill="none" stroke-width="1.5"/>
        <line x1="132" y1="155" x2="112" y2="190" stroke="#1a1a1a" stroke-width="1.2"/>
        <line x1="168" y1="155" x2="188" y2="190" stroke="#1a1a1a" stroke-width="1.2"/>
        """
        side = """
        <ellipse cx="450" cy="200" rx="20" ry="55" stroke="#1a1a1a" fill="none" stroke-width="2"/>
        <ellipse cx="450" cy="200" rx="13" ry="38" stroke="#1a1a1a" fill="none" stroke-width="1.5"/>
        <ellipse cx="450" cy="152" rx="8" ry="12" stroke="#cc8800" fill="none" stroke-width="1.5"/>
        """
        top = """
        <ellipse cx="750" cy="200" rx="55" ry="20" stroke="#1a1a1a" fill="none" stroke-width="2"/>
        <ellipse cx="750" cy="200" rx="38" ry="13" stroke="#1a1a1a" fill="none" stroke-width="1.5"/>
        <ellipse cx="750" cy="200" rx="18" ry="8" stroke="#cc8800" fill="none" stroke-width="1.5"/>
        """
    elif "necklace" in jtype or "pendant" in jtype:
        front = """
        <rect x="80" y="120" width="140" height="160" rx="8" stroke="#1a1a1a" fill="none" stroke-width="2"/>
        <ellipse cx="150" cy="180" rx="30" ry="30" stroke="#cc8800" fill="none" stroke-width="1.5"/>
        <line x1="150" y1="120" x2="150" y2="80" stroke="#1a1a1a" stroke-width="2"/>
        """
        side = """
        <rect x="430" y="120" width="40" height="160" rx="4" stroke="#1a1a1a" fill="none" stroke-width="2"/>
        <ellipse cx="450" cy="180" rx="10" ry="10" stroke="#cc8800" fill="none" stroke-width="1.5"/>
        """
        top = """
        <rect x="700" y="170" width="100" height="60" rx="8" stroke="#1a1a1a" fill="none" stroke-width="2"/>
        <ellipse cx="750" cy="200" rx="22" ry="15" stroke="#cc8800" fill="none" stroke-width="1.5"/>
        """
    elif "earring" in jtype:
        front = """
        <line x1="150" y1="100" x2="150" y2="140" stroke="#1a1a1a" stroke-width="2"/>
        <ellipse cx="150" cy="200" rx="40" ry="60" stroke="#1a1a1a" fill="none" stroke-width="2"/>
        <ellipse cx="150" cy="190" rx="18" ry="18" stroke="#cc8800" fill="none" stroke-width="1.5"/>
        """
        side = """
        <line x1="450" y1="100" x2="450" y2="140" stroke="#1a1a1a" stroke-width="2"/>
        <ellipse cx="450" cy="200" rx="12" ry="60" stroke="#1a1a1a" fill="none" stroke-width="2"/>
        """
        top = """
        <ellipse cx="750" cy="200" rx="40" ry="15" stroke="#1a1a1a" fill="none" stroke-width="2"/>
        <ellipse cx="750" cy="200" rx="18" ry="7" stroke="#cc8800" fill="none" stroke-width="1.5"/>
        """
    else:
        front = """
        <ellipse cx="150" cy="200" rx="70" ry="70" stroke="#1a1a1a" fill="none" stroke-width="2.5"/>
        <ellipse cx="150" cy="200" rx="55" ry="55" stroke="#1a1a1a" fill="none" stroke-width="1.5"/>
        """
        side = """
        <ellipse cx="450" cy="200" rx="22" ry="70" stroke="#1a1a1a" fill="none" stroke-width="2"/>
        <ellipse cx="450" cy="200" rx="16" ry="55" stroke="#1a1a1a" fill="none" stroke-width="1.5"/>
        """
        top = """
        <ellipse cx="750" cy="200" rx="70" ry="22" stroke="#1a1a1a" fill="none" stroke-width="2"/>
        <ellipse cx="750" cy="200" rx="55" ry="16" stroke="#1a1a1a" fill="none" stroke-width="1.5"/>
        """

    return f"""<svg viewBox="0 0 900 400" xmlns="http://www.w3.org/2000/svg" font-family="monospace">
  <rect width="900" height="400" fill="white"/>
  <text x="450" y="28" font-size="13" fill="#111111" text-anchor="middle" font-weight="bold">
    JEWELLERY TECHNICAL WIREFRAME — {jewellery_type.upper()} | {metal} {karat} | Style: {style}
  </text>
  <line x1="300" y1="40" x2="300" y2="370" stroke="#cccccc" stroke-dasharray="6,4" stroke-width="1"/>
  <line x1="600" y1="40" x2="600" y2="370" stroke="#cccccc" stroke-dasharray="6,4" stroke-width="1"/>
  <text x="150" y="58" font-size="11" fill="{label_color}" text-anchor="middle" font-weight="bold">FRONT VIEW</text>
  <text x="450" y="58" font-size="11" fill="{label_color}" text-anchor="middle" font-weight="bold">SIDE VIEW</text>
  <text x="750" y="58" font-size="11" fill="{label_color}" text-anchor="middle" font-weight="bold">TOP VIEW</text>
  {front}{side}{top}
  <line x1="80" y1="310" x2="220" y2="310" stroke="{dim_color}" stroke-width="0.8" marker-end="url(#arrow)" marker-start="url(#arrow)"/>
  <text x="150" y="325" font-size="9" fill="{dim_color}" text-anchor="middle">~{width_mm}mm</text>
  <text x="245" y="210" font-size="9" fill="{dim_color}" text-anchor="start">~{height_mm}mm</text>
  <rect x="20" y="350" width="10" height="10" fill="none" stroke="#1a1a1a" stroke-width="1.5"/>
  <text x="35" y="360" font-size="9" fill="{label_color}">Metal outline</text>
  <rect x="110" y="350" width="10" height="10" fill="none" stroke="{stone_color}" stroke-width="1.5"/>
  <text x="125" y="360" font-size="9" fill="{label_color}">Stone / Setting</text>
  <text x="700" y="360" font-size="9" fill="{label_color}" text-anchor="middle">Complexity: {complexity}</text>
  <defs>
    <marker id="arrow" markerWidth="6" markerHeight="6" refX="3" refY="3" orient="auto">
      <path d="M0,0 L6,3 L0,6 Z" fill="{dim_color}"/>
    </marker>
  </defs>
</svg>"""


def _generate_cad_instructions(spec: dict, jewellery_type: str) -> str:
    """Generate plain-text CAD instruction sheet from spec."""
    dims = spec.get("dimensions", {})
    components = spec.get("components", [])
    stones = spec.get("stones", [])

    lines = [
        "=" * 60,
        "  JEWELLERY CAD INSTRUCTION SHEET",
        "  Generated by Manappuram AI PoC",
        "=" * 60,
        "",
        f"TYPE          : {jewellery_type}",
        f"METAL         : {spec.get('metal', 'N/A')} {spec.get('karat', '')}",
        f"STYLE         : {spec.get('design_style', 'N/A')}",
        f"SYMMETRY      : {spec.get('symmetry', 'N/A')}",
        f"COMPLEXITY    : {spec.get('complexity_level', 'N/A')}",
        "",
        "--- ESTIMATED DIMENSIONS ---",
        f"  Width  : ~{dims.get('estimated_width_mm', 'N/A')} mm",
        f"  Height : ~{dims.get('estimated_height_mm', 'N/A')} mm",
        f"  Depth  : ~{dims.get('estimated_depth_mm', 'N/A')} mm",
        f"  Band   : ~{dims.get('band_thickness_mm', 'N/A')} mm thick",
        "",
        "--- COMPONENTS ---",
    ]

    for i, comp in enumerate(components, 1):
        lines += [
            f"  {i}. {comp.get('name', 'Component')}",
            f"     Shape    : {comp.get('shape', 'N/A')}",
            f"     Material : {comp.get('material', 'N/A')}",
            f"     Finish   : {comp.get('finish', 'N/A')}",
            f"     Position : {comp.get('position', 'N/A')}",
            "",
        ]

    if stones and stones[0].get("type", "None") != "None":
        lines += ["--- STONES ---"]
        for i, stone in enumerate(stones, 1):
            lines += [
                f"  {i}. {stone.get('type', 'N/A')}",
                f"     Cut     : {stone.get('cut', 'N/A')}",
                f"     Setting : {stone.get('setting', 'N/A')}",
                f"     Count   : {stone.get('count', 'N/A')}",
                f"     Size    : ~{stone.get('estimated_size_mm', 'N/A')} mm",
                "",
            ]

    lines += [
        "--- CAD DESIGNER NOTES ---",
        textwrap.fill(
            spec.get("cad_notes", "No additional notes."),
            width=56, initial_indent="  ", subsequent_indent="  "
        ),
        "",
        "--- TIME ESTIMATE ---",
        f"  Without AI reference : ~{spec.get('estimated_cad_hours_without_ai', 6)} hours",
        f"  With AI reference    : ~{spec.get('estimated_cad_hours_with_ai_reference', 1.5)} hours",
        f"  Time saved           : ~{spec.get('estimated_cad_hours_without_ai', 6) - spec.get('estimated_cad_hours_with_ai_reference', 1.5):.1f} hours",
        "",
        "=" * 60,
        "  Import SVG into Rhino / MatrixGold / Fusion 360",
        "  as a reference layer to start CAD design.",
        "=" * 60,
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API — called from app.py
# ---------------------------------------------------------------------------

def generate_3d_from_image(
    pil_image: Image.Image,
    output_basename: str,
) -> Tuple[Optional[Path], Optional[Path], Optional[Path], Optional[str]]:
    """
    Main function. Does 4 things in order:
      0. VALIDATE — check image actually contains jewellery
      1. ANALYSE  — extract structured JSON spec using Claude
      2. DRAW     — generate SVG wireframe using Claude
      3. WRITE    — save SVG, TXT, JSON files

    Returns:
      (svg_path, txt_path, json_path, None)      on success
      (None, None, None, error_message)           on failure
    """

    # ── Step 0: Validate image contains jewellery ──────────────────────────
    # FIX: Validation is correctly placed INSIDE this function
    # so it runs BEFORE any API calls or file generation
    from .inspiration import extract_style_tags

    print("[three_d] Step 0: Validating image contains jewellery...")
    validation = extract_style_tags(pil_image)

    if validation and validation[0].startswith("REJECTED:"):
        reason = validation[0].replace("REJECTED:", "").strip()
        print(f"[three_d] Validation failed: {reason}")
        return None, None, None, (
            f"⚠️ Non-jewellery image detected: {reason}. "
            f"Please upload a jewellery photo."
        )

    if validation and validation[0].startswith("Error:"):
        return None, None, None, validation[0]

    # ── Step 1: Check API key ───────────────────────────────────────────────
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        return None, None, None, (
            "ANTHROPIC_API_KEY is not set. "
            "Add it to your .env file: ANTHROPIC_API_KEY=sk-ant-..."
        )

    THREE_D_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # ── Step 2: Analyse image → structured JSON spec ───────────────────
        print("[three_d] Step 1: Analysing jewellery image with Claude Vision...")
        raw_spec = _call_claude_vision(pil_image, ANALYSIS_PROMPT, max_tokens=2048)
        spec = _parse_json_safe(raw_spec)
        jewellery_type = spec.get(
            "jewellery_type",
            output_basename.replace("_model", "").title()
        )

        # ── Step 3: Generate SVG wireframe ──────────────────────────────────
        print("[three_d] Step 2: Generating technical wireframe SVG...")
        try:
            svg_raw = _call_claude_vision(pil_image, SVG_PROMPT, max_tokens=3000)
            svg_match = re.search(r"<svg.*</svg>", svg_raw, re.DOTALL)
            svg_content = svg_match.group() if svg_match else None
        except Exception as svg_err:
            print(f"[three_d] SVG generation failed, using fallback: {svg_err}")
            svg_content = None

        if not svg_content:
            svg_content = _generate_fallback_svg(jewellery_type, spec)

        # ── Step 4: Generate CAD instruction sheet ──────────────────────────
        print("[three_d] Step 3: Generating CAD instruction sheet...")
        cad_instructions = _generate_cad_instructions(spec, jewellery_type)

        # ── Step 5: Save all 3 files ────────────────────────────────────────
        svg_path  = THREE_D_OUTPUT_DIR / f"{output_basename}_wireframe.svg"
        txt_path  = THREE_D_OUTPUT_DIR / f"{output_basename}_cad_instructions.txt"
        json_path = THREE_D_OUTPUT_DIR / f"{output_basename}_spec.json"

        svg_path.write_text(svg_content, encoding="utf-8")
        txt_path.write_text(cad_instructions, encoding="utf-8")
        json_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")

        print(f"[three_d] Done. Files saved to {THREE_D_OUTPUT_DIR}")
        return svg_path, txt_path, json_path, None

    except Exception as e:
        return None, None, None, f"3D analysis failed: {e}"