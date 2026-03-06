from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
import torch

from .config import CATALOG_IMAGES_DIR, EMBEDDING_MODEL_NAME

_device = "cuda" if torch.cuda.is_available() else "cpu"
_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=_device)
    return _model


def load_image(image_path: Path) -> Image.Image:
    img = Image.open(image_path).convert("RGB")
    return img


def encode_images(image_paths: List[Path]) -> np.ndarray:
    model = get_model()
    images = [load_image(p) for p in image_paths]
    embeddings = model.encode(images, batch_size=16, convert_to_numpy=True, show_progress_bar=True)
    return embeddings


def encode_single_image(image: Image.Image) -> np.ndarray:
    model = get_model()
    emb = model.encode([image], convert_to_numpy=True)[0]
    return emb


def list_catalog_images() -> List[Tuple[str, Path]]:
    """Returns list of (product_id, image_path) assuming <product_id>.jpg naming."""
    image_paths = []
    for p in CATALOG_IMAGES_DIR.glob("*.*"):
        product_id = p.stem  # filename without extension
        image_paths.append((product_id, p))
    return sorted(image_paths, key=lambda x: x[0])
