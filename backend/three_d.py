from pathlib import Path
from typing import Tuple, Optional

import numpy as np
from PIL import Image
from huggingface_hub import login
from transformers import pipeline

from .config import IMAGE_TO_3D_MODEL, HF_TOKEN, THREE_D_OUTPUT_DIR

_image_to_3d_pipeline = None


def _get_pipeline():
    global _image_to_3d_pipeline
    if _image_to_3d_pipeline is None:
        if HF_TOKEN:
            login(token=HF_TOKEN)
        _image_to_3d_pipeline = pipeline(
            "image-to-3d",
            model=IMAGE_TO_3D_MODEL,
            trust_remote_code=True,
        )
    return _image_to_3d_pipeline


def generate_3d_from_image(
    pil_image: Image.Image,
    output_basename: str,
) -> Tuple[Optional[Path], Optional[str]]:
    """
    Returns (mesh_path, error_message). Mesh is typically a .ply or .glb file.
    """
    THREE_D_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pipe = _get_pipeline()

    image = np.array(pil_image, dtype=np.float32) / 255.0
    try:
        result = pipe("", image)
        mesh_path = THREE_D_OUTPUT_DIR / f"{output_basename}.ply"
        pipe.save_ply(result, str(mesh_path))
        return mesh_path, None
    except Exception as e:
        return None, str(e)
