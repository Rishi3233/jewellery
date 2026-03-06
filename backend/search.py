from dataclasses import dataclass
from typing import List

import faiss
import numpy as np
import pandas as pd
from PIL import Image

from .config import (
    FAISS_INDEX_PATH,
    EMBEDDINGS_NPY_PATH,      # kept for possible future use
    PRODUCT_IDS_NPY_PATH,
    CATALOG_METADATA_PATH,
    CATALOG_IMAGES_DIR,
)
from .embeddings import encode_single_image


@dataclass
class ProductResult:
    product_id: str
    score: float
    metadata: dict
    image_path: str


_index = None
_product_ids = None
_metadata_df = None


def _load_index():
    """
    Lazy-load FAISS index, product_ids array, and metadata dataframe.
    """
    global _index, _product_ids, _metadata_df
    if _index is None:
        _index = faiss.read_index(str(FAISS_INDEX_PATH))
        _product_ids = np.load(PRODUCT_IDS_NPY_PATH)
        _metadata_df = pd.read_csv(CATALOG_METADATA_PATH)


def _get_metadata(product_id: str) -> dict:
    """
    Look up metadata row(s) for the given product_id.

    Assumes:
      - metadata.csv has a column "Product ID" (adjust if you use a different name).
    """
    # Adjust column name here if your metadata uses "product_id" instead
    rows = _metadata_df[_metadata_df["product_id"] == product_id]
    if rows.empty:
        return {}
    # If there are multiple rows (multiple images per product), just take the first here.
    # You can later extend this to return all variants if needed.
    return rows.iloc[0].to_dict()


def search_similar(image: Image.Image, top_k: int = 3) -> List[ProductResult]:
    """
    Encode the query image, perform similarity search in FAISS,
    and return top_k ProductResult entries.

    Assumes:
      - PRODUCT_IDS_NPY_PATH stores one product_id per indexed embedding.
      - metadata.csv has "Product ID" and "image_file" columns.
      - All images are stored under CATALOG_IMAGES_DIR.
    """
    _load_index()

    emb = encode_single_image(image)
    emb = emb.astype("float32")[None, :]
    faiss.normalize_L2(emb)
    scores, indices = _index.search(emb, top_k)
    scores = scores[0]
    indices = indices[0]

    results: List[ProductResult] = []
    for score, idx in zip(scores, indices):
        if idx < 0:
            continue

        pid = str(_product_ids[idx])
        meta = _get_metadata(pid)

        # Build image path from metadata's image_file; fall back to product_id.jpg
        image_file = meta.get("image_file", None)
        if image_file:
            image_path = str(CATALOG_IMAGES_DIR / image_file)
        else:
            image_path = str(CATALOG_IMAGES_DIR / f"{pid}.jpg")

        results.append(
            ProductResult(
                product_id=str(pid),
                score=float(score),
                metadata=meta,
                image_path=image_path,
            )
        )

    return results
