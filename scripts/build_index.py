

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


def load_items_from_metadata(product_id_column: str = "product_id") -> list[tuple[str, Path]]:
    """
    Load (product_id, image_path) pairs from metadata.csv.

    FIX: Default column name changed from "Product ID" to "product_id"
         to match the column name used in search.py (_get_metadata function).
         Both files now consistently use "product_id" (lowercase, underscore).

    Args:
        product_id_column: Column name in metadata.csv containing the product ID.
                           Defaults to "product_id".

    Returns:
        List of (product_id_string, image_path) tuples.
    """
    if not CATALOG_METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata file not found at {CATALOG_METADATA_PATH}")

    df = pd.read_csv(CATALOG_METADATA_PATH)

    if product_id_column not in df.columns:
        raise KeyError(
            f"Expected column '{product_id_column}' in metadata.csv. "
            f"Available columns: {list(df.columns)}"
        )

    if "image_file" not in df.columns:
        raise KeyError(
            f"Expected column 'image_file' in metadata.csv. "
            f"Available columns: {list(df.columns)}"
        )

    records: list[tuple[str, Path]] = []
    missing_images = []

    for _, row in df.iterrows():
        pid = str(row[product_id_column])
        img_file = str(row["image_file"])
        img_path = CATALOG_IMAGES_DIR / img_file

        if not img_path.exists():
            missing_images.append(str(img_path))
            continue  # skip missing images instead of crashing

        records.append((pid, img_path))

    if missing_images:
        print(f"WARNING: Skipped {len(missing_images)} missing image(s):")
        for p in missing_images[:10]:  # show max 10
            print(f"  - {p}")
        if len(missing_images) > 10:
            print(f"  ... and {len(missing_images) - 10} more.")

    return records


def main():
    parser = argparse.ArgumentParser(description="Build FAISS index for catalog images.")
    parser.add_argument(
        "--product-id-column",
        type=str,
        # FIX: default changed from "Product ID" to "product_id"
        # so it matches the column name used in search.py
        default="product_id",
        help="Column name in metadata.csv that contains the product ID. Default: product_id",
    )
    args = parser.parse_args()

    print(f"Loading catalog items from metadata (column: '{args.product_id_column}')...")
    items = load_items_from_metadata(product_id_column=args.product_id_column)
    product_ids = [pid for pid, _ in items]
    paths = [p for _, p in items]

    print(f"Found {len(paths)} valid images.")
    if not paths:
        print("No images found. Please check metadata.csv and image_file paths.")
        return

    print("Encoding images with CLIP model (this may take a few minutes)...")
    embeddings = encode_images(paths)
    dim = embeddings.shape[1]
    print(f"Embedding dimension: {dim}")

    # Save embeddings and product IDs
    EMBEDDINGS_NPY_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_NPY_PATH, embeddings)
    np.save(PRODUCT_IDS_NPY_PATH, np.array(product_ids))
    print(f"Saved embeddings → {EMBEDDINGS_NPY_PATH}")
    print(f"Saved product IDs → {PRODUCT_IDS_NPY_PATH}")

    # Build FAISS index with L2-normalised vectors for cosine similarity
    print("Building FAISS index...")
    faiss.normalize_L2(embeddings.astype("float32"))
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype("float32"))

    FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    print(f"Index built with {index.ntotal} vectors → {FAISS_INDEX_PATH}")
    print("Done! Run `streamlit run app.py` to start the app.")


if __name__ == "__main__":
    main()