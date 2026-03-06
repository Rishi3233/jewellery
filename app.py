import io
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

from backend.config import (
    CATALOG_METADATA_PATH,
    THREE_D_OUTPUT_DIR,
    CATALOG_IMAGES_DIR,
)

from backend.config import CATALOG_IMAGES_DIR
from backend.search import search_similar
from backend.three_d import generate_3d_from_image
from backend.inspiration import extract_style_tags, generate_concepts_from_image


st.set_page_config(
    page_title="Manappuram Jewellery – Image AI PoC",
    layout="wide",
)


@st.cache_data
def load_metadata() -> pd.DataFrame:
    if CATALOG_METADATA_PATH.exists():
        df = pd.read_csv(CATALOG_METADATA_PATH)

        #Clean price: remove commas/spaces and convert to float
        if "price" in df.columns:
            df["price"] = (
                df["price"]
                .astype(str)
                .str.replace(",","", regex=False)
                .str.strip()
            )
            df["price"] = pd.to_numeric(df["price"], errors="coerce")

        return df
    # fallback dummy df
    return pd.DataFrame(columns=["product_id", "category", "metal", "karat", "gross_weight", "stone_weight", "price", "image_file"])


def show_product_card(res, width=200):
    col = st.container()
    with col:
        st.image(res.image_path, width=width, caption=f"{res.product_id}")
        price = res.metadata.get("price", "NA")
        category = res.metadata.get("category", "")
        st.markdown(f"Category: {category}")
        st.markdown(f"Price: {price}")
        st.markdown(f"Similarity: {res.score:.3f}")


#def get_uploaded_image(file) -> Image.Image:
    #image_bytes = file.read()
    #image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    #return image
def get_uploaded_image(file) -> Image.Image:
    # Streamlit's UploadedFile exposes getvalue(), which is safer than read()
    image_bytes = file.getvalue()
    if not image_bytes:
        raise ValueError("Uploaded file is empty or could not be read.")
    image = Image.open(io.BytesIO(image_bytes))
    return image.convert("RGB")

st.sidebar.title("Manappuram Jewellery – PoC")
st.sidebar.markdown("Image-based search, 3D model generation, and design inspiration.")

# Global filters (for later use, currently not wired into search)
meta_df = load_metadata()
categories = sorted(meta_df["category"].dropna().unique().tolist()) if not meta_df.empty else []
selected_categories = st.sidebar.multiselect("Category filter", options=categories, default=categories)

st.sidebar.markdown("---")
st.sidebar.markdown("Powered by open-source ViT & HF 3D models.")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Exact/Similar Search", "Visual Discovery", "Image → 3D CAD", "Design Inspiration"]
)

# ----- Tab 1 -----
with tab1:
    st.header("Find Matching Designs")
    st.write("Upload a customer photo or sketch to find exact and visually similar designs from the catalog.")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        uploaded = st.file_uploader("Upload image or sketch", type=["jpg", "jpeg", "png"], key="search_upload")
        noisy_bg = st.checkbox("Photo taken in-store (background cleanup)")
        search_btn = st.button("Search Catalog")

    with col_right:
        # show either newly uploaded image or last_image from session_state
        image = None
        if uploaded is not None:
            image = get_uploaded_image(uploaded)
            st.session_state["last_image"] = image
        elif "last_image" in st.session_state:
            image = st.session_state["last_image"]
        if image is not None:
            st.image(image, caption="Reference image", use_column_width=True)
    results = None

    # Case 1: user clicked Search in this tab
    if search_btn and uploaded is not None:
        with st.spinner("Searching catalog..."):
            results = search_similar(image, top_k=5)
        st.session_state["last_results"] = results

    # Case 2: user came from Visual Discovery / Inspiration tab
    elif "last_results" in st.session_state:
        results = st.session_state["last_results"]

    # Render results if we have any
    if results:
        st.subheader("Best Match")
        best = results[0]
        best_cols = st.columns([1, 2])
        with best_cols[0]:
            show_product_card(best, width=260)
        with best_cols[1]:
            st.markdown("**Product Details**")
            for k, v in best.metadata.items():
                st.markdown(f"- {k}: {v}")

        st.subheader("Visually Similar Designs")
        grid_cols = st.columns(4)
        for i, res in enumerate(results[1:]):
            with grid_cols[i % 4]:
                show_product_card(res, width=200)
        else:
            st.info("Upload an image here or use 'Find Similar' from other tabs to see results.")

# ----- Tab 2 -----
with tab2:
    st.header("Visual Discovery")
    st.write("Browse the catalog visually and quickly find similar designs.")

    if meta_df.empty:
        st.info("No metadata found. Populate data/catalog/metadata.csv to enable visual discovery.")
    else:
        # basic filters
        col_filters = st.columns(3)
        with col_filters[0]:
            cat_filter = st.selectbox("Category", options=["All"] + categories, index=0)
        with col_filters[1]:
            min_price, max_price = float(meta_df["price"].min() or 0), float(meta_df["price"].max() or 0)
            price_range = st.slider("Price range", min_value=min_price, max_value=max_price, value=(min_price, max_price))
        with col_filters[2]:
            search_text = st.text_input("Search by name/ID contains", "")

        df = meta_df.copy()
        if cat_filter != "All":
            df = df[df["category"] == cat_filter]
        df = df[(df["price"] >= price_range[0]) & (df["price"] <= price_range[1])]
        if search_text:
            df = df[df["product_id"].astype(str).str.contains(search_text)]

        st.write(f"{len(df)} products")

        # pagination
        page_size = 12
        page = st.number_input("Page", min_value=1, value=1, step=1)
        start = (page - 1) * page_size
        end = start + page_size
        page_df = df.iloc[start:end]

        grid_cols = st.columns(4)
        for i, row in page_df.iterrows():
            with grid_cols[i % 4]:
                img_file = row.get("image_file", f"{row['product_id']}.jpg")
                img_path = str(CATALOG_IMAGES_DIR / img_file)
                st.image(img_path, width=200)
                st.markdown(f"**{row['category']}**")
                st.caption(f"ID: {row['product_id']} | ₹{row['price']}")
                if st.button("Find Similar", key=f"find_similar_{row['product_id']}_{img_file}"):
                    # Load image and run search
                    try:
                        ref_img = Image.open(img_path).convert("RGB")
                        st.session_state["last_image"] = ref_img
                        with st.spinner("Searching similar designs..."):
                            res = search_similar(ref_img, top_k=5)
                        st.session_state["last_results"] = res
                        st.success("Similar designs fetched. Switch to 'Exact/Similar Search' tab to view.")
                    except Exception as e:
                        st.error(f"Error loading image: {e}")

# ----- Tab 3 -----
with tab3:
    st.header("Image → 3D CAD-Ready Model")
    st.write("Upload a reference image to generate a 3D mesh that can be refined in jewellery CAD tools.")

    col_left, col_right = st.columns([1, 1])
    with col_left:
        uploaded_3d = st.file_uploader("Upload jewellery reference image", type=["jpg", "jpeg", "png"], key="three_d_upload")
        jewellery_type = st.selectbox("Jewellery type", options=["Ring", "Pendant", "Bangle", "Earring", "Necklace"])
        generate_btn = st.button("Generate 3D Model")

        if st.button("Use last search image") and "last_image" in st.session_state:
            uploaded_3d_image = st.session_state["last_image"]
        else:
            uploaded_3d_image = None

    with col_right:
        if uploaded_3d:
            img = get_uploaded_image(uploaded_3d)
            st.image(img, caption="Reference image", use_column_width=True)
            st.session_state["three_d_image"] = img
        elif "three_d_image" in st.session_state:
            st.image(st.session_state["three_d_image"], caption="Reference image (from previous)", use_column_width=True)

    if generate_btn:
        if uploaded_3d:
            img = get_uploaded_image(uploaded_3d)
        else:
            img = st.session_state.get("three_d_image", None)

        if img is None:
            st.warning("Please upload an image or use the last search image.")
        else:
            basename = f"{jewellery_type.lower()}_model"
            with st.spinner("Generating 3D model (this may take some time)..."):
                mesh_path, err = generate_3d_from_image(img, output_basename=basename)
            if err:
                st.error(f"3D generation failed: {err}")
            elif mesh_path:
                st.success("3D model generated.")
                st.write("Download the mesh and import into Rhino/Matrix/Fusion for CAD refinement.")
                st.download_button(
                    "Download 3D mesh (.ply)",
                    data=open(mesh_path, "rb").read(),
                    file_name=mesh_path.name,
                    mime="application/octet-stream",
                )

# ----- Tab 4 -----
with tab4:
    st.header("Design Inspiration – New Designs from Images")
    st.write("Use a customer image to generate style tags and concept variations.")

    col_left, col_right = st.columns([1, 1])
    with col_left:
        uploaded_insp = st.file_uploader("Upload customer reference image", type=["jpg", "jpeg", "png"], key="insp_upload")
        style_notes = st.text_input("Style notes (optional)", placeholder="Floral halo ring with emerald center, antique finish")
        insp_btn = st.button("Generate Concepts")

    with col_right:
        if uploaded_insp:
            img_insp = get_uploaded_image(uploaded_insp)
            st.image(img_insp, caption="Customer reference", use_column_width=True)
            st.session_state["insp_image"] = img_insp

    if insp_btn and uploaded_insp:
        img_insp = get_uploaded_image(uploaded_insp)
        with st.spinner("Generating style tags and concepts..."):
            tags = extract_style_tags(img_insp)
            concepts = generate_concepts_from_image(img_insp)

        st.subheader("Detected Style Attributes")
        st.write(", ".join(tags))

        st.subheader("AI Concept Variations")
        concept_cols = st.columns(4)
        for i, c in enumerate(concepts):
            with concept_cols[i % 4]:
                st.markdown(f"**{c.title}**")
                st.caption(c.description)
                st.image(c.reference_image_path, width=200, caption=f"Based on {c.reference_product_id}")
                if st.button("Find similar in catalog", key=f"insp_find_{i}"):
                    # reuse that product image in search
                    try:
                        ref_img = Image.open(c.reference_image_path).convert("RGB")
                        st.session_state["last_image"] = ref_img
                        with st.spinner("Searching similar designs..."):
                            res = search_similar(ref_img, top_k=5)
                        st.session_state["last_results"] = res
                        st.success("Similar designs fetched. Switch to 'Exact/Similar Search' tab to view.")
                    except Exception as e:
                        st.error(f"Error loading reference image: {e}")
                if st.button("Send to 3D generation", key=f"insp_3d_{i}"):
                    try:
                        img = Image.open(c.reference_image_path).convert("RGB")
                        st.session_state["three_d_image"] = img
                        st.success("Reference sent to 3D tab. Open 'Image → 3D CAD' to generate mesh.")
                    except Exception as e:
                        st.error(f"Error sending image: {e}")
