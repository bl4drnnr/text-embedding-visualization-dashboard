import streamlit as st
import pandas as pd
import numpy as np

from text_embedding_visualization_dashboard.frontend.utils import (
    apply_dimensionality_reduction,
    get_embeddings,
    create_embeddings,
)
from text_embedding_visualization_dashboard.frontend.visualizations import plot_reduced_embeddings
from text_embedding_visualization_dashboard.vector_db import VectorDB
from text_embedding_visualization_dashboard.embeddings import Embeddings

if "is_processing" not in st.session_state:
    st.session_state.is_processing = False

AVAILABLE_MODELS = {
    "all-mpnet-base-v2": {"speed": 2800, "size": "420 MB"},
    "multi-qa-mpnet-base-dot-v1": {"speed": 2800, "size": "420 MB"},
    "all-distilroberta-v1": {"speed": 4000, "size": "290 MB"},
    "all-MiniLM-L12-v2": {"speed": 7500, "size": "120 MB"},
    "multi-qa-distilbert-cos-v1": {"speed": 4000, "size": "250 MB"},
    "all-MiniLM-L6-v2": {"speed": 14200, "size": "80 MB"},
    "multi-qa-MiniLM-L6-cos-v1": {"speed": 14200, "size": "80 MB"},
    "paraphrase-multilingual-mpnet-base-v2": {"speed": 2500, "size": "970 MB"},
    "paraphrase-albert-small-v2": {"speed": 5000, "size": "43 MB"},
    "paraphrase-multilingual-MiniLM-L12-v2": {"speed": 7500, "size": "420 MB"},
    "paraphrase-MiniLM-L3-v2": {"speed": 19000, "size": "61 MB"},
    "distiluse-base-multilingual-cased-v1": {"speed": 4000, "size": "480 MB"},
    "distiluse-base-multilingual-cased-v2": {"speed": 4000, "size": "480 MB"},
}

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

model_option = st.sidebar.selectbox(
    "Choose an embedding model",
    options=list(AVAILABLE_MODELS.keys()),
    index=list(AVAILABLE_MODELS.keys()).index("all-MiniLM-L6-v2"),
    help="Select the model to use for generating embeddings. Speed indicates sentences per second, size indicates model size.",
    disabled=st.session_state.is_processing
)

model_specs = AVAILABLE_MODELS[model_option]
st.sidebar.markdown(f"""
**Model Specifications:**
- Speed: {model_specs['speed']} sentences/sec
- Size: {model_specs['size']}
""")

if "current_model" not in st.session_state:
    st.session_state.current_model = model_option
    st.session_state.embeddings_instance = Embeddings(db, model_name=model_option)
    st.session_state.is_processing = False
elif st.session_state.current_model != model_option:
    st.session_state.current_model = model_option
    st.session_state.embeddings_instance = Embeddings(db, model_name=model_option)

dataset_option = st.selectbox(
    "Choose a data source", 
    ["Upload your own data", "Existing dataset"],
    disabled=st.session_state.is_processing
)

uploaded_file = None
dataset_name = None
if dataset_option == "Upload your own data":
    uploaded_file = st.file_uploader(
        "Upload CSV file", 
        type=["csv"],
        disabled=st.session_state.is_processing
    )
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Available columns in your dataset:")
        st.write(df.columns.tolist())
        
        uploaded_file.seek(0)
        
        label_column = st.text_input(
            "Enter the name of your label column",
            value="label",
            help="Specify which column contains the labels for your texts. This column should contain categorical values that will be used to color the visualization."
        )
        
        if st.button("Process Dataset", disabled=st.session_state.is_processing):
            if label_column not in df.columns:
                st.error(f"Column '{label_column}' not found in the dataset. Please check the column name and try again.")
            else:
                dataset_name = create_embeddings(st.session_state.embeddings_instance, uploaded_file, label_column)
                st.success("Dataset processed successfully!")
                st.experimental_set_query_params(dataset_option="Existing dataset")
                st.rerun()


if dataset_option == "Existing dataset":
    collections = [col.name for col in db.get_all_collections()]
    dataset_name = st.selectbox(
        "Choose a dataset", 
        collections,
        disabled=st.session_state.is_processing
    )

dimensionality_reduction_option = st.sidebar.selectbox(
    "Choose a dimensionality reduction technique",
    ["t-SNE", "UMAP", "PaCMAP", "TriMAP"],
    disabled=st.session_state.is_processing
)


if "dr_params" not in st.session_state:
    st.session_state.dr_params = {
        "t-SNE": {"perplexity": 5, "max_iter": 300},
        "UMAP": {"n_neighbors": 5, "min_dist": 0.1},
        "PaCMAP": {"n_neighbors": 5},
        "TriMAP": {"n_neighbors": 5},
    }

if "current_reduction" not in st.session_state:
    st.session_state.current_reduction = None

embeddings, labels = None, None
if dataset_name:
    embeddings, labels = get_embeddings(db, dataset_name)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Compute Reduction")

if "dataset_size" not in st.session_state:
    st.session_state.dataset_size = None

if labels is not None:
    st.session_state.dataset_size = len(labels)

dr_params = st.session_state.dr_params
dataset_size = st.session_state.dataset_size

if dataset_size is not None:
    if dimensionality_reduction_option == "t-SNE":
        dr_params["t-SNE"] = {
            "perplexity": st.sidebar.slider(
                "perplexity", 
                5, 
                min(50, dataset_size - 1), 
                dr_params["t-SNE"]["perplexity"],
                disabled=st.session_state.is_processing
            ),
            "max_iter": st.sidebar.slider(
                "iterations", 
                250, 
                1000, 
                dr_params["t-SNE"]["max_iter"],
                disabled=st.session_state.is_processing
            ),
            "n_components": 3,
        }

    if dimensionality_reduction_option == "UMAP":
        dr_params["UMAP"] = {
            "n_neighbors": st.sidebar.slider(
                "n_neighbors", 
                5, 
                min(100, dataset_size - 1), 
                dr_params["UMAP"]["n_neighbors"],
                disabled=st.session_state.is_processing
            ),
            "min_dist": st.sidebar.slider(
                "min_dist", 
                0.01, 
                0.99, 
                dr_params["UMAP"]["min_dist"], 
                step=0.01,
                disabled=st.session_state.is_processing
            ),
            "n_components": 3,
        }

    if dimensionality_reduction_option == "PaCMAP":
        dr_params["PaCMAP"] = {
            "n_neighbors": st.sidebar.slider(
                "n_neighbors", 
                5, 
                min(50, dataset_size - 1), 
                dr_params["PaCMAP"]["n_neighbors"],
                disabled=st.session_state.is_processing
            ),
            "n_components": 3,
        }

    if dimensionality_reduction_option == "TriMAP":
        dr_params["TriMAP"] = {
            "n_neighbors": st.sidebar.slider(
                "n_neighbors", 
                5, 
                min(50, dataset_size - 1), 
                dr_params["TriMAP"]["n_neighbors"],
                disabled=st.session_state.is_processing
            ),
            "n_components": 3,
        }

    if st.sidebar.button("Run Dimensionality Reduction", disabled=st.session_state.is_processing):
        st.session_state.is_processing = True
        try:
            with st.spinner(f"Computing {dimensionality_reduction_option} projection..."):
                st.session_state.current_reduction = apply_dimensionality_reduction(
                    embeddings, dimensionality_reduction_option, dr_params[dimensionality_reduction_option]
                )
        finally:
            st.session_state.is_processing = False

if embeddings is not None and st.session_state.current_reduction is not None:
    tab2D, tab3D = st.tabs(["2D", "3D"])

    with tab2D:
        fig = plot_reduced_embeddings(st.session_state.current_reduction, labels, dimensionality_reduction_option, type="2D")
        st.plotly_chart(fig, use_container_width=True, key="2D")

    with tab3D:
        fig3D = plot_reduced_embeddings(st.session_state.current_reduction, labels, dimensionality_reduction_option, type="3D")
        st.plotly_chart(fig3D, use_container_width=True, key="3D")
