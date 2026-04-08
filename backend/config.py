import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[1]

CATALOG_IMAGES_DIR    = BASE_DIR / "data" / "catalog" / "images"
CATALOG_METADATA_PATH = BASE_DIR / "data" / "catalog" / "metadata.csv"
FAISS_INDEX_PATH      = BASE_DIR / "models" / "faiss_index.bin"
EMBEDDINGS_NPY_PATH   = BASE_DIR / "models" / "image_embeddings.npy"
PRODUCT_IDS_NPY_PATH  = BASE_DIR / "models" / "product_ids.npy"

# Output directories
THREE_D_OUTPUT_DIR      = BASE_DIR / "data" / "outputs" / "three_d"
# FIX: Added separate folder for inspiration/Novita generated images
INSPIRATION_OUTPUT_DIR  = BASE_DIR / "data" / "outputs" / "inspiration"

# CLIP model for image embeddings
EMBEDDING_MODEL_NAME = "sentence-transformers/clip-ViT-B-32"

# Image-to-3D stub reference
IMAGE_TO_3D_MODEL = "tencent/Hunyuan3D-2.1"

# API Keys — loaded from .env file
HF_TOKEN          = os.getenv("HF_TOKEN", None)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", None)
NOVITA_API_KEY    = os.getenv("NOVITA_API_KEY", None)