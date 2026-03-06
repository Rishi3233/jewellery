import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[1]

CATALOG_IMAGES_DIR = BASE_DIR / "data" / "catalog" / "images"
CATALOG_METADATA_PATH = BASE_DIR / "data" / "catalog" / "metadata.csv"
FAISS_INDEX_PATH = BASE_DIR / "models" / "faiss_index.bin"
EMBEDDINGS_NPY_PATH = BASE_DIR / "models" / "image_embeddings.npy"
PRODUCT_IDS_NPY_PATH = BASE_DIR / "models" / "product_ids.npy"

THREE_D_OUTPUT_DIR = BASE_DIR / "data" / "outputs" / "three_d"

# Open-source ViT model for image embeddings
EMBEDDING_MODEL_NAME = "sentence-transformers/clip-ViT-B-32"  # open model [web:77]

# Open-source image-to-3D model (LGM) for HF pipeline [web:47][web:81]
IMAGE_TO_3D_MODEL = "tencent/Hunyuan3D-2.1"

# Hugging Face token (if needed for gated models)
HF_TOKEN = os.getenv("HF_TOKEN", None)