import argparse
from pathlib import Path

import numpy as np
import faiss
import pandas as pd

from backend.embeddings import encode_images
from backend.config import (
    EMBEDDINGS_NPY_PATH,
    PRODUCT_IDS_NPY_PATH,
    FAISS_INDEX_PATH,
    CATALOG_METADATA_PATH,
    CATALOG_IMAGES_DIR,
)


def load_items_from_metadata(product_id_column: str = "Product ID") -> list[tuple[str, Path]]:
    """
    Load (product_id, image_path) pairs from metadata.csv.

    Assumes:
      - metadata.csv has a column for product ID (default: "Product ID" – adjust if needed).
      - metadata.csv has a column "image_file" containing the image filename.
      - All images are stored under CATALOG_IMAGES_DIR.
    """
    if not CATALOG_METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata file not found at {CATALOG_METADATA_PATH}")

    df = pd.read_csv(CATALOG_METADATA_PATH)

    if product_id_column not in df.columns:
        raise KeyError(f"Expected column '{product_id_column}' in metadata.csv")

    if "image_file" not in df.columns:
        raise KeyError("Expected column 'image_file' in metadata.csv")

    records: list[tuple[str, Path]] = []
    for _, row in df.iterrows():
        pid = str(row[product_id_column])
        img_file = str(row["image_file"])
        img_path = CATALOG_IMAGES_DIR / img_file
        records.append((pid, img_path))

    return records


def main():
    parser = argparse.ArgumentParser(description="Build FAISS index for catalog images.")
    parser.add_argument(
        "--product-id-column",
        type=str,
        default="Product ID",
        help="Column name in metadata.csv that contains the product ID.",
    )
    args = parser.parse_args()

    print("Loading catalog items from metadata...")
    items = load_items_from_metadata(product_id_column=args.product_id_column)
    product_ids = [pid for pid, _ in items]
    paths = [p for _, p in items]

    print(f"Found {len(paths)} images.")
    if not paths:
        print("No images found. Please check metadata and image_file paths.")
        return

    print("Encoding images...")
    embeddings = encode_images(paths)
    dim = embeddings.shape[1]

    # Save embeddings and product IDs
    EMBEDDINGS_NPY_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_NPY_PATH, embeddings)
    np.save(PRODUCT_IDS_NPY_PATH, np.array(product_ids))

    # Build FAISS index
    print("Building FAISS index...")
    index = faiss.IndexFlatIP(dim)
    # Normalize for cosine similarity (cosine similarity via inner product)
    faiss.normalize_L2(embeddings)
    index.add(embeddings.astype("float32"))

    FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    print(f"Index built and saved to {FAISS_INDEX_PATH}")


if __name__ == "__main__":
    main()
