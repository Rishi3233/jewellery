```markdown
# Manappuram Jewellery – Image Search & 3D PoC

This project is a Proof of Concept (PoC) for **image‑based jewellery search and 3D model generation** for Manappuram Jewellery.

Using a Streamlit UI, store teams and customers can:

- Upload photos or sketches to:
  - Find **exact** and **visually similar** designs from the product catalog.
  - Explore catalog visually (Visual Discovery).
- Generate a **3D mesh** (stubbed in this PoC) from a reference image for CAD workflows.
- Use customer images as inspiration to generate **design concepts**.

---

## 1. Project Structure

```text
manappuram_jewellery_poc/
├── app.py                     # Main Streamlit app
├── backend/
│   ├── __init__.py
│   ├── config.py             # Paths, model names, HF token
│   ├── embeddings.py         # Image embedding utilities
│   ├── search.py             # FAISS similarity search
│   ├── three_d.py            # 3D mesh generation (stub)
│   └── inspiration.py        # Simple design inspiration logic
├── data/
│   ├── catalog/
│   │   ├── images/           # Product images
│   │   └── metadata.csv      # Catalog metadata (see schema below)
│   └── outputs/
│       └── three_d/          # Generated 3D meshes (.ply)
├── models/
│   ├── faiss_index.bin       # FAISS index for image similarity
│   ├── image_embeddings.npy  # Pre‑computed image embeddings
│   └── product_ids.npy       # Product IDs aligned to embeddings
├── scripts/
│   └── build_index.py        # Script to build embeddings + FAISS index
├── requirements.txt
└── README.md
```

---

## 2. Prerequisites

- Python 3.9+ (recommended)
- Git
- (Optional) GPU if you later replace the 3D stub with a real image‑to‑3D model.

---

## 3. Setup

### 3.1 Clone the repository

```bash
git clone <your_repo_url>.git
cd manappuram_jewellery_poc
```

### 3.2 Create and activate virtual environment

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3.3 Install dependencies

```bash
pip install -r requirements.txt
```

### 3.4 (Optional) Hugging Face token

If you later plug in a real image‑to‑3D model from Hugging Face, create a `.env` file in the project root:

```env
HF_TOKEN=your_huggingface_token_here
```

`config.py` automatically loads this value.

---

## 4. Catalog Data Preparation

Place your catalog data under `data/catalog/`:

1. **Images**:  
   - Directory: `data/catalog/images/`  
   - Files: one or more images per product, e.g.:
     - `GRGPL_1916.jpg`
     - `GRGPL_1916_side_view.jpg`

2. **Metadata**:  
   - File: `data/catalog/metadata.csv`  
   - Required columns:

```csv
product_id,category,metal,karat,gross_weight,stone_weight,price,image_file
GRGPL_1916,Finger Ring,Gold,22K,3.00 gm,NA,31537,GRGPL_1916.jpg
GRGPL_1916,Finger Ring,Gold,22K,3.00 gm,NA,31537,GRGPL_1916_side_view.jpg
...
```

Notes:

- `price` should be numeric (no currency symbols; commas will be cleaned in code).
- `image_file` is the **exact file name** in `data/catalog/images/`.

---

## 5. Build Embeddings and FAISS Index

From the project root:

```bash
python scripts/build_index.py --product-id-column product_id
```

This will:

- Encode all images using an open‑source ViT/CLIP model from `sentence-transformers`.
- Save embeddings to `models/image_embeddings.npy`.
- Save product IDs to `models/product_ids.npy`.
- Build a FAISS index saved as `models/faiss_index.bin`.

Whenever you change `metadata.csv` or add/remove images, run the above command again to rebuild the index.

---

## 6. Running the Streamlit App

From the project root (with the virtual environment active):

```bash
streamlit run app.py
```

Streamlit will show a local URL (usually `http://localhost:8501`). Open it in your browser.

---

## 7. App Features (Tabs)

### 7.1 Exact/Similar Search

- Upload a customer photo or sketch.
- App shows:
  - **Best Match** from the catalog.
  - A grid of **visually similar designs**.

You can also land here after clicking **“Find Similar”** in other tabs; results are shared via `st.session_state`.

### 7.2 Visual Discovery

- Browse products visually with:
  - Category filter.
  - Price range slider.
  - Simple ID filter.
- Click **“Find Similar”** on any product to:
  - Run similarity search using that image.
  - Store results for viewing in the Exact/Similar Search tab.

### 7.3 Image → 3D CAD (Stub)

- Upload an image and generate a **placeholder 3D mesh (.ply)**.
- This PoC stub creates a simple cube mesh so the full flow (upload → generate → download) is wired; you can replace the stub with a real image‑to‑3D backend later.

### 7.4 Design Inspiration

- Upload a customer reference image.
- PoC extracts simple style tags (placeholder) and:
  - Shows a few “concept” cards based on similar catalog items.
  - Lets you:
    - **Find similar in catalog**, or
    - **Send to 3D** to reuse the concept image in the 3D tab.

---

## 8. Handling `ModuleNotFoundError: No module named 'backend'`

If you see an error like:

```text
ModuleNotFoundError: No module named 'backend'
```

do the following:

1. Make sure you are running commands from the **project root** (folder containing `app.py` and `backend/`), not inside `scripts/`:

```bash
cd manappuram_jewellery_poc
```

2. Ensure `backend/__init__.py` exists (even if empty).

3. **On Windows PowerShell**, set `PYTHONPATH` to include the current directory, then run the script:

```powershell
$env:PYTHONPATH="."
python scripts/build_index.py --product-id-column product_id
```

You can use the same approach when running Streamlit if needed:

```powershell
$env:PYTHONPATH="."
streamlit run app.py
```

This tells Python explicitly to treat the project root as part of the import path so `backend.*` modules can be imported correctly.

---

## 9. Future Extensions

- Swap the 3D stub in `backend/three_d.py` with:
  - A local TRELLIS or Hunyuan3D‑2 backend, or
  - An external CAD/3D service.
- Improve style extraction and concept generation with:
  - Fine‑tuned visual classifiers.
  - An LLM backend (e.g., Groq or HF chat models).
- Add authentication, logging, and analytics for a production‑ready deployment.

---

## 10. License

NexTurn Proprietary and should be shared and used only within NexTurn Organization.
```