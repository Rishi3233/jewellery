import warnings
warnings.filterwarnings("ignore")

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
import streamlit as st

from backend.config import (
    CATALOG_METADATA_PATH,
    THREE_D_OUTPUT_DIR,
    CATALOG_IMAGES_DIR,
)
from backend.search import search_similar
from backend.three_d import generate_3d_from_image
# FIX: Added run_novita_generation import — was missing before
from backend.inspiration import (
    extract_style_tags,
    generate_concepts_from_image,
    run_novita_generation,
)


st.set_page_config(
    page_title="Manappuram Jewellery – Image AI PoC",
    layout="wide",
)


# ---------------------------------------------------------------------------
# LOGIN SYSTEM
# ---------------------------------------------------------------------------

USERS = {
    "customer1": {"password": "cust123",   "role": "customer", "name": "Customer"},
    "designer1": {"password": "design123", "role": "designer", "name": "CAD Designer"},
    "admin":     {"password": "admin123",  "role": "admin",    "name": "Admin"},
}

ROLE_PERMISSIONS = {
    "customer": {
        "can_search": True, "can_browse": True, "can_3d": True,
        "can_download_svg": False, "can_download_txt": False,
        "can_download_json": False, "can_see_inspiration": True,
    },
    "designer": {
        "can_search": True, "can_browse": True, "can_3d": True,
        "can_download_svg": True, "can_download_txt": True,
        "can_download_json": False, "can_see_inspiration": True,
    },
    "admin": {
        "can_search": True, "can_browse": True, "can_3d": True,
        "can_download_svg": True, "can_download_txt": True,
        "can_download_json": True, "can_see_inspiration": True,
    },
}


def show_login_page():
    st.title("Manappuram Jewellery – Image AI")
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.subheader("🔐 Login")
        username = st.text_input("Username", placeholder="Enter username")
        password = st.text_input("Password", type="password", placeholder="Enter password")
        login_btn = st.button("Login", use_container_width=True)
        if login_btn:
            if username in USERS and USERS[username]["password"] == password:
                st.session_state["logged_in"] = True
                st.session_state["username"] = username
                st.session_state["role"] = USERS[username]["role"]
                st.session_state["name"] = USERS[username]["name"]
                st.rerun()
            else:
                st.error("❌ Wrong username or password. Please try again.")
        st.markdown("---")



def logout():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


# ---------------------------------------------------------------------------
# Login check
# ---------------------------------------------------------------------------

if "logged_in" not in st.session_state or not st.session_state["logged_in"]:
    show_login_page()
    st.stop()

current_role = st.session_state["role"]
current_name = st.session_state["name"]
perms = ROLE_PERMISSIONS[current_role]
ROLE_COLORS = {"customer": "🟢", "designer": "🔵", "admin": "🔴"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_data
def load_metadata() -> pd.DataFrame:
    if CATALOG_METADATA_PATH.exists():
        df = pd.read_csv(CATALOG_METADATA_PATH)
        if "price" in df.columns:
            df["price"] = (
                df["price"].astype(str)
                .str.replace(",", "", regex=False).str.strip()
            )
            df["price"] = pd.to_numeric(df["price"], errors="coerce")
        return df
    return pd.DataFrame(columns=["product_id", "category", "metal", "karat",
                                  "gross_weight", "stone_weight", "price", "image_file"])


def show_product_card(res, width=200):
    with st.container():
        try:
            st.image(res.image_path, width=width, caption=f"{res.product_id}")
        except Exception:
            st.warning(f"Image not found: {res.product_id}")
        st.markdown(f"Category: {res.metadata.get('category', '')}")
        st.markdown(f"Price: ₹{res.metadata.get('price', 'NA')}")
        st.markdown(f"Match Quality: {res.score * 100:.1f}%")


def get_uploaded_image(file) -> "Image.Image | None":
    if file is None:
        return None
    try:
        image_bytes = file.getvalue()
    except Exception:
        return None
    if not image_bytes:
        return None
    try:
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

meta_df = load_metadata()
categories = sorted(meta_df["category"].dropna().unique().tolist()) if not meta_df.empty else []

st.sidebar.title("Manappuram Jewellery – PoC")
st.sidebar.markdown("---")
st.sidebar.markdown(f"**{ROLE_COLORS[current_role]} Logged in as:**")
st.sidebar.markdown(f"**{current_name}**")
st.sidebar.markdown(f"Role: `{current_role.upper()}`")
st.sidebar.markdown("---")
st.sidebar.markdown("**Your Access:**")
st.sidebar.markdown(f"{'✅' if perms['can_search'] else '❌'} Image Search")
st.sidebar.markdown(f"{'✅' if perms['can_browse'] else '❌'} Visual Discovery")
st.sidebar.markdown(f"{'✅' if perms['can_3d'] else '❌'} 3D CAD Reference")
st.sidebar.markdown(f"{'✅' if perms['can_download_svg'] else '❌'} Download SVG")
st.sidebar.markdown(f"{'✅' if perms['can_download_txt'] else '❌'} Download Instructions")
st.sidebar.markdown(f"{'✅' if perms['can_download_json'] else '❌'} Download JSON Spec")
st.sidebar.markdown("---")
selected_categories = st.sidebar.multiselect("Category filter", options=categories, default=categories)
st.sidebar.markdown("---")
if st.sidebar.button("🚪 Logout"):
    logout()


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs(
    ["Exact/Similar Search", "Visual Discovery", "Image → 3D CAD", "Design Inspiration"]
)

# ── Tab 1 ─────────────────────────────────────────────────────────────────────
with tab1:
    st.header("Find Matching Designs")
    st.write("Upload a customer photo or sketch to find exact and visually similar designs from the catalog.")

    col_left, col_right = st.columns([1, 1])
    with col_left:
        uploaded = st.file_uploader("Upload image or sketch", type=["jpg", "jpeg", "png"], key="search_upload")
        noisy_bg = st.checkbox("Photo taken in-store (background cleanup)")
        search_btn = st.button("Search Catalog")

    with col_right:
        image = None
        if uploaded is not None:
            image = get_uploaded_image(uploaded)
            if image is None:
                st.error("Could not read the file. Please upload a valid JPG or PNG.")
            else:
                st.session_state["last_image"] = image
        elif "last_image" in st.session_state:
            image = st.session_state["last_image"]
        if image is not None:
            st.image(image, caption="Reference image", use_container_width=True)

    results = None
    if search_btn:
        if image is None:
            st.warning("Please upload a valid image first.")
        else:
            # ── Jewellery validation before catalog search ──────────────────
            with st.spinner("Validating image..."):
                validation = extract_style_tags(image)
            if validation and validation[0].startswith("REJECTED:"):
                reason = validation[0].replace("REJECTED:", "").strip()
                st.error(f"⚠️ Non-jewellery image detected: {reason}")
                st.warning("Please upload a jewellery photo to search the catalog.")
                st.stop()
            # ── Validation passed — proceed with catalog search ─────────────
            with st.spinner("Searching catalog..."):
                all_results = search_similar(image, top_k=10)
            if selected_categories:
                results = [r for r in all_results
                           if r.metadata.get("category", "") in selected_categories][:5]
            else:
                results = all_results[:5]
            st.session_state["last_results"] = results
    elif "last_results" in st.session_state:
        results = st.session_state["last_results"]

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

# ── Tab 2 ─────────────────────────────────────────────────────────────────────
with tab2:
    st.header("Visual Discovery")
    st.write("Browse the catalog visually and quickly find similar designs.")

    if meta_df.empty:
        st.info("No metadata found. Populate data/catalog/metadata.csv to enable visual discovery.")
    else:
        col_filters = st.columns(3)
        with col_filters[0]:
            cat_filter = st.selectbox("Category", options=["All"] + categories, index=0)
        with col_filters[1]:
            min_price = float(meta_df["price"].min() or 0)
            max_price = float(meta_df["price"].max() or 0)
            if min_price == max_price:
                max_price = min_price + 1.0
            price_range = st.slider("Price range", min_value=min_price,
                                    max_value=max_price, value=(min_price, max_price))
        with col_filters[2]:
            search_text = st.text_input("Search by name/ID contains", "")

        df = meta_df.copy()
        if cat_filter != "All":
            df = df[df["category"] == cat_filter]
        df = df[(df["price"] >= price_range[0]) & (df["price"] <= price_range[1])]
        if search_text:
            df = df[df["product_id"].astype(str).str.contains(search_text, case=False)]

        st.write(f"{len(df)} products")
        page_size = 12
        page = st.number_input("Page", min_value=1, value=1, step=1)
        start = (page - 1) * page_size
        page_df = df.iloc[start: start + page_size]

        grid_cols = st.columns(4)
        for i, row in page_df.iterrows():
            with grid_cols[i % 4]:
                img_file = row.get("image_file", f"{row['product_id']}.jpg")
                img_path = str(CATALOG_IMAGES_DIR / img_file)
                try:
                    pil_img = Image.open(img_path).convert("RGB")
                    st.image(pil_img, width=200)
                except Exception:
                    st.warning(f"⚠️ Bad image: {img_file}")
                    continue
                st.markdown(f"**{row['category']}**")
                st.caption(f"ID: {row['product_id']} | ₹{row['price']}")
                if st.button("Find Similar", key=f"find_similar_{row['product_id']}_{img_file}"):
                    try:
                        ref_img = Image.open(img_path).convert("RGB")
                        st.session_state["last_image"] = ref_img
                        with st.spinner("Searching similar designs..."):
                            res = search_similar(ref_img, top_k=5)
                        st.session_state["last_results"] = res
                        st.success("Similar designs fetched. Switch to 'Exact/Similar Search' tab to view.")
                    except Exception as e:
                        st.error(f"Error loading image: {e}")

# ── Tab 3 ─────────────────────────────────────────────────────────────────────
with tab3:
    st.header("Image → 3D CAD Reference")
    st.write("Upload a jewellery photo to generate a technical wireframe, dimensions, and CAD instruction sheet.")

    if current_role == "customer":
        st.info("💡 You can view the 3D analysis on screen. Downloads available for designers and admins.")
    elif current_role == "designer":
        st.info("💡 You can view and download the SVG wireframe and CAD instruction sheet.")
    elif current_role == "admin":
        st.info("💡 Full access — view and download all files including JSON spec.")

    col_left, col_right = st.columns([1, 1])
    with col_left:
        uploaded_3d = st.file_uploader(
            "Upload jewellery reference image", type=["jpg", "jpeg", "png"], key="three_d_upload"
        )
        jewellery_type = st.selectbox(
            "Jewellery type hint",
            options=["Ring", "Pendant", "Bangle", "Earring", "Necklace", "Nosepin"]
        )
        generate_btn = st.button("🔬 Analyse & Generate CAD Reference")
        use_last_btn = st.button("Use last search image")
        if use_last_btn and "last_image" in st.session_state:
            st.session_state["three_d_image"] = st.session_state["last_image"]
            st.success("Last search image loaded.")

    with col_right:
        if uploaded_3d:
            img_3d = get_uploaded_image(uploaded_3d)
            if img_3d is None:
                st.error("Could not read uploaded file. Please upload a valid JPG or PNG.")
            else:
                st.image(img_3d, caption="Reference image", use_container_width=True)
                st.session_state["three_d_image"] = img_3d
        elif "three_d_image" in st.session_state:
            st.image(st.session_state["three_d_image"],
                     caption="Reference image (from previous)", use_container_width=True)

    if generate_btn:
        img_3d = get_uploaded_image(uploaded_3d) if uploaded_3d else st.session_state.get("three_d_image")
        if img_3d is None:
            st.warning("Please upload a valid image or use the last search image.")
        else:
            basename = f"{jewellery_type.lower()}_model"
            with st.spinner("Claude is analysing and generating CAD reference... (15–30 seconds)"):
                svg_path, txt_path, json_path, err = generate_3d_from_image(img_3d, output_basename=basename)

            if err:
                st.error(f"Analysis failed: {err}")
            else:
                st.success("✅ CAD Reference generated successfully!")
                st.subheader("Technical Wireframe (3 Views)")
                st.caption("Front View · Side View · Top View")
                try:
                    svg_content = svg_path.read_text(encoding="utf-8")
                    st.components.v1.html(svg_content, height=420, scrolling=False)
                except Exception as e:
                    st.warning(f"Could not render SVG: {e}")

                if json_path and json_path.exists():
                    st.subheader("3D Specification")
                    try:
                        spec_data = json.loads(json_path.read_text(encoding="utf-8"))
                        spec_cols = st.columns(3)
                        with spec_cols[0]:
                            st.metric("Jewellery Type", spec_data.get("jewellery_type", "N/A"))
                            st.metric("Metal", f"{spec_data.get('metal', 'N/A')} {spec_data.get('karat', '')}")
                            st.metric("Complexity", spec_data.get("complexity_level", "N/A"))
                        with spec_cols[1]:
                            dims = spec_data.get("dimensions", {})
                            st.metric("Width", f"~{dims.get('estimated_width_mm', 'N/A')} mm")
                            st.metric("Height", f"~{dims.get('estimated_height_mm', 'N/A')} mm")
                            st.metric("Depth", f"~{dims.get('estimated_depth_mm', 'N/A')} mm")
                        with spec_cols[2]:
                            hours_before = spec_data.get("estimated_cad_hours_without_ai", 6)
                            hours_after = spec_data.get("estimated_cad_hours_with_ai_reference", 1.5)
                            saved = hours_before - hours_after
                            st.metric("CAD Hours (without AI)", f"~{hours_before}h")
                            st.metric("CAD Hours (with AI ref)", f"~{hours_after}h")
                            st.metric("⏱ Time Saved", f"~{saved:.1f}h", delta=f"-{saved:.1f}h")
                        if spec_data.get("cad_notes"):
                            st.info(f"**Designer Note:** {spec_data['cad_notes']}")
                    except Exception as e:
                        st.warning(f"Could not parse spec: {e}")

                st.subheader("Download Files")
                dl_cols = st.columns(3)
                with dl_cols[0]:
                    if perms["can_download_svg"]:
                        if svg_path and svg_path.exists():
                            st.download_button("⬇ Download SVG Wireframe",
                                               data=svg_path.read_bytes(),
                                               file_name=svg_path.name, mime="image/svg+xml")
                    else:
                        st.caption("🔒 SVG download — Designer/Admin only")
                with dl_cols[1]:
                    if perms["can_download_txt"]:
                        if txt_path and txt_path.exists():
                            st.download_button("⬇ Download CAD Instructions",
                                               data=txt_path.read_bytes(),
                                               file_name=txt_path.name, mime="text/plain")
                    else:
                        st.caption("🔒 Instructions — Designer/Admin only")
                with dl_cols[2]:
                    if perms["can_download_json"]:
                        if json_path and json_path.exists():
                            st.download_button("⬇ Download JSON Spec",
                                               data=json_path.read_bytes(),
                                               file_name=json_path.name, mime="application/json")
                    else:
                        st.caption("🔒 JSON download — Admin only")

# ── Tab 4 ─────────────────────────────────────────────────────────────────────
with tab4:
    st.header("Design Inspiration – Professional AI Studio")
    st.write("Upload a jewellery image to generate AI concept variations and find catalog matches.")

    col_left, col_right = st.columns([1, 1])
    with col_left:
        uploaded_insp = st.file_uploader(
            "Upload customer reference image", type=["jpg", "jpeg", "png"], key="insp_upload"
        )
        style_notes = st.text_input(
            "Style notes (optional)",
            placeholder="Floral halo ring with emerald center, antique finish"
        )
        negative_notes = st.text_input(
            "Elements to remove (optional)",
            placeholder="green stones, heavy shadows..."
        )
        # FIX: Added rendering mode selector — required for Novita API routing
        st.markdown("### Rendering Mode")
        action_mode = st.radio(
            "What would you like the AI to do?",
            ["🎨 Edit Existing Design", "✏️ Render a Sketch"]
        )
        insp_btn = st.button("Generate Concepts")

    with col_right:
        if uploaded_insp:
            img_insp = get_uploaded_image(uploaded_insp)
            if img_insp:
                st.image(img_insp, caption="Customer reference", use_container_width=True)
                st.session_state["insp_image"] = img_insp

    if insp_btn:
        # Get image from upload or session state
        img_insp = None
        if uploaded_insp:
            img_insp = get_uploaded_image(uploaded_insp)
        elif "insp_image" in st.session_state:
            img_insp = st.session_state["insp_image"]

        if img_insp is None:
            st.warning("Please upload a valid image first.")
        else:
            # Stage 1: Claude analysis
            with st.spinner("Claude is analysing the jewellery..."):
                tags = extract_style_tags(img_insp, style_notes)

            # Case 1: API error
            if tags and tags[0].startswith("Error:"):
                st.error(tags[0])

            # Case 2: Not jewellery — stop everything
            elif tags and tags[0].startswith("REJECTED:"):
                st.warning("⚠️ Non-Jewellery Image Detected")
                st.info(f"Claude Analysis: {tags[0].replace('REJECTED:', '').strip()}")
                st.error("Please upload a valid jewellery reference image.")
                st.stop()

            # Case 3: Valid jewellery — full pipeline
            else:
                st.subheader("Claude's Analysis")
                clean_analysis = tags[0].replace("#", "").replace("**", "").strip()
                st.info(f"✨ {clean_analysis}")

                # Stage 2: Novita AI image generation
                st.subheader("Novita AI Generated Vision")
                with st.spinner("Novita AI is rendering the new design (~15-30 seconds)..."):
                    generated_img_path = run_novita_generation(
                        img_insp, tags[0], action_mode, negative_notes
                    )

                if generated_img_path and Path(generated_img_path).exists():
                    st.image(generated_img_path, use_container_width=True,
                             caption=f"Mode: {action_mode}")
                else:
                    st.warning(
                        "⚠️ Novita render not available. "
                        "Check NOVITA_API_KEY in .env file."
                    )

                # Stage 3: FAISS catalog search
                st.subheader("Closest Matches in Existing Catalog")
                with st.spinner("Searching FAISS catalog..."):
                    # If Novita made a new image — search using it
                    if generated_img_path and Path(generated_img_path).exists():
                        search_img = Image.open(generated_img_path).convert("RGB")
                    else:
                        search_img = img_insp
                    concepts = generate_concepts_from_image(search_img, tags[0])

                if not concepts:
                    st.warning("No mathematically close matches found in the current catalog.")
                else:
                    concept_cols = st.columns(min(4, len(concepts)))
                    for i, c in enumerate(concepts):
                        with concept_cols[i % len(concept_cols)]:
                            st.markdown(f"**{c.title}**")
                            st.caption(c.description)

                            # Show Novita generated image on first card if available
                            # FIX: c.generated_image_path now exists in CatalogItem
                            if c.generated_image_path and Path(c.generated_image_path).exists():
                                try:
                                    st.image(c.generated_image_path, width=200,
                                             caption="🤖 AI Generated")
                                except Exception:
                                    pass

                            # Always show catalog reference image
                            try:
                                st.image(c.reference_image_path, width=200,
                                         caption=f"📦 ID: {c.reference_product_id}")
                            except Exception:
                                st.warning(f"Image not available: {c.reference_product_id}")

                            if st.button("Find similar in catalog", key=f"insp_find_{i}"):
                                try:
                                    ref_img = Image.open(c.reference_image_path).convert("RGB")
                                    st.session_state["last_image"] = ref_img
                                    with st.spinner("Searching..."):
                                        res = search_similar(ref_img, top_k=5)
                                    st.session_state["last_results"] = res
                                    st.success("Switch to 'Exact/Similar Search' tab to view.")
                                except Exception as e:
                                    st.error(f"Error: {e}")

                            if st.button("Send to 3D generation", key=f"insp_3d_{i}"):
                                try:
                                    img = Image.open(c.reference_image_path).convert("RGB")
                                    st.session_state["three_d_image"] = img
                                    st.success("Sent to 3D tab.")
                                except Exception as e:
                                    st.error(f"Error: {e}")