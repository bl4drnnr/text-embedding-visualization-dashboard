import streamlit as st

from text_embedding_visualization_dashboard.frontend.utils import (
    apply_dimensionality_reduction,
    get_embeddings,
    create_embeddings,
)
from text_embedding_visualization_dashboard.frontend.visualizations import plot_reduced_embeddings
from text_embedding_visualization_dashboard.vector_db import VectorDB


db = VectorDB()

st.set_page_config(page_title="Text Embedding Visualization Dashboard", page_icon="📊", layout="wide")

st.title("Text Embedding Visualization Dashboard")
st.markdown("""
This application allows you to visualize text embeddings using different dimensionality reduction techniques:
- UMAP (Uniform Manifold Approximation and Projection)
- t-SNE (t-Distributed Stochastic Neighbor Embedding)
- PaCMAP (Pairwise Controlled Manifold Approximation)
- TriMAP (Triple Manifold Approximation and Projection)
""")


st.sidebar.header("Settings")

dataset_option = st.selectbox("Choose a data source", ["Upload your own data", "Existing dataset"])

uploaded_file = None
dataset_name = None
if dataset_option == "Upload your own data":
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        dataset_name = create_embeddings(db, uploaded_file)


if dataset_option == "Existing dataset":
    collections = [col.name for col in db.get_all_collections()]
    dataset_name = st.selectbox("Choose a dataset", collections)

dimensionality_reduction_option = st.sidebar.selectbox(
    "Choose a dimensionality reduction technique",
    ["t-SNE", "UMAP", "PaCMAP", "TriMAP"],
)


if "dr_params" not in st.session_state:
    st.session_state.dr_params = {
        "t-SNE": {"perplexity": 5, "max_iter": 300},
        "UMAP": {"n_neighbors": 5, "min_dist": 0.1},
        "PaCMAP": {"n_neighbors": 5},
        "TriMAP": {"n_neighbors": 5},
    }

embeddings, labels = None, None
if dataset_name:
    embeddings, labels = get_embeddings(db, dataset_name)

if "dataset_size" not in st.session_state:
    st.session_state.dataset_size = None

if labels is not None:
    st.session_state.dataset_size = len(labels)

dr_params = st.session_state.dr_params
dataset_size = st.session_state.dataset_size

# Dimensionality reduction settings
if dataset_size is not None:
    if dimensionality_reduction_option == "t-SNE":
        dr_params["t-SNE"] = {
            "perplexity": st.sidebar.slider(
                "perplexity", 5, min(50, dataset_size - 1), dr_params["t-SNE"]["perplexity"]
            ),
            "max_iter": st.sidebar.slider("iterations", 250, 1000, dr_params["t-SNE"]["max_iter"]),
            "n_components": 3,
        }

    if dimensionality_reduction_option == "UMAP":
        dr_params["UMAP"] = {
            "n_neighbors": st.sidebar.slider(
                "n_neighbors", 5, min(100, dataset_size - 1), dr_params["UMAP"]["n_neighbors"]
            ),
            "min_dist": st.sidebar.slider("min_dist", 0.01, 0.99, dr_params["UMAP"]["min_dist"], step=0.01),
            "n_components": 3,
        }

    if dimensionality_reduction_option == "PaCMAP":
        dr_params["PaCMAP"] = {
            "n_neighbors": st.sidebar.slider(
                "n_neighbors", 5, min(50, dataset_size - 1), dr_params["PaCMAP"]["n_neighbors"]
            ),
            "n_components": 3,
        }

    if dimensionality_reduction_option == "TriMAP":
        dr_params["TriMAP"] = {
            "n_neighbors": st.sidebar.slider(
                "n_neighbors", 5, min(50, dataset_size - 1), dr_params["TriMAP"]["n_neighbors"]
            ),
            "n_components": 3,
        }

# Visualizations
if embeddings is not None:
    tab2D, tab3D = st.tabs(["2D", "3D"])

    with st.spinner(f"Computing {dimensionality_reduction_option} projection..."):
        embeddings_reduced = apply_dimensionality_reduction(
            embeddings, dimensionality_reduction_option, dr_params[dimensionality_reduction_option]
        )

        with tab2D:
            fig = plot_reduced_embeddings(embeddings_reduced, labels, dimensionality_reduction_option, type="2D")
            st.plotly_chart(fig, use_container_width=True, key="2D")

        with tab3D:
            fig3D = plot_reduced_embeddings(embeddings_reduced, labels, dimensionality_reduction_option, type="3D")
            st.plotly_chart(fig3D, use_container_width=True, key="3D")
