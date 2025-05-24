from typing import Dict, Tuple

import streamlit as st
import pandas as pd
import umap
import trimap
import pacmap
from sklearn.manifold import TSNE
import numpy as np

from text_embedding_visualization_dashboard.vector_db import VectorDB
from text_embedding_visualization_dashboard.embeddings import Embeddings


def create_embeddings(embeddings_instance: Embeddings, uploaded_file, label_column: str = "label") -> str | None:
    """
    Creates embeddings from uploaded CSV file and adds them to the vector database.

    Parameters:
    embeddings_instance : Embeddings
        The embeddings instance to use for generating embeddings.
    uploaded_file : file object
        CSV file uploaded by the user. Must contain:
        - a 'text' column with the text data
        - a column specified by label_column containing the labels (if not named 'label')
        - optionally an 'id' column for custom document IDs (if not present, will be auto-generated)
    label_column : str
        Name of the column containing the labels. Defaults to "label". If provided, this name will be used
        as the key in the metadata dictionary.

    Returns:
    str or None
        The name of the created collection, or None if there was an error.
    """

    df = pd.read_csv(uploaded_file)

    if "text" not in df.columns:
        st.error("The CSV file must contain a 'text' column.")
        return None

    if label_column not in df.columns:
        st.error(f"The CSV file must contain a '{label_column}' column.")
        return None

    labels = df[label_column].fillna("unknown").astype(str).tolist()
    texts = df["text"].tolist()
    collection_name = uploaded_file.name[:-4]
    
    if "id" not in df.columns:
        ids = [f"doc_{i}" for i in range(len(texts))]
    else:
        ids = df["id"].tolist()
    
    metadatas = [{label_column: label} for label in labels]
    
    embeddings_instance.batch_process_texts(
        texts=texts,
        collection_name=collection_name,
        metadatas=metadatas,
        ids=ids,
        batch_size=5000
    )

    return collection_name


def get_embeddings(db: VectorDB, dataset_name: str) -> Tuple[np.ndarray, list[str]]:
    """
    Returns embeddings and corresponding labels from the database.

    Parameters:
    db : VectorDB
        The vector database instance for retrieving embeddings.
    dataset_name : str
        Name of the collection to retrieve embeddings from.

    Returns:
    tuple
        (embeddings, labels): The retrieved embeddings as numpy array and corresponding labels.
    """

    db_collection = db.get_all_items_from_collection(dataset_name, include=["embeddings", "metadatas"])

    embeddings = np.array(db_collection["embeddings"])

    metadatas = db_collection["metadatas"]
    
    if metadatas and len(metadatas) > 0:
        label_key = next(iter(metadatas[0].keys()))
        labels = [metadata.get(label_key) for metadata in metadatas]
    else:
        labels = []

    return embeddings, labels


def apply_dimensionality_reduction(embeddings: np.ndarray, method: str, params: Dict[str, int | float]) -> np.ndarray:
    """
    Apply dimensionality reduction to embedding vectors using the specified method.

    Parameters:
    embeddings : np.ndarray
        The high-dimensional embedding vectors to reduce.
    method : str
        The dimensionality reduction method to use. Should be one of:
        'UMAP', 't-SNE', 'PaCMAP', or 'TriMAP'.
    params : Dict[str, int | float]
        Parameters for the dimensionality reduction method.

    Returns:
    np.ndarray
        The reduced embeddings with shape (n_samples, n_components).
    """

    random_state = 42
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        if method == "UMAP":
            status_text.text("Initializing UMAP...")
            progress_bar.progress(10)
            reducer = umap.UMAP(
                n_neighbors=params["n_neighbors"],
                min_dist=params["min_dist"],
                n_components=params["n_components"],
                random_state=random_state,
            )
            status_text.text("Computing UMAP projection...")
            progress_bar.progress(30)
            reduced = reducer.fit_transform(embeddings)
            progress_bar.progress(100)

        elif method == "t-SNE":
            status_text.text("Initializing t-SNE...")
            progress_bar.progress(10)
            reducer = TSNE(
                n_components=params["n_components"],
                perplexity=params["perplexity"],
                max_iter=params["max_iter"],
                random_state=random_state,
            )
            status_text.text("Computing t-SNE projection (this may take a while)...")
            progress_bar.progress(30)
            reduced = reducer.fit_transform(embeddings)
            progress_bar.progress(100)

        elif method == "PaCMAP":
            status_text.text("Initializing PaCMAP...")
            progress_bar.progress(10)
            reducer = pacmap.PaCMAP(
                n_neighbors=params["n_neighbors"], 
                n_components=params["n_components"], 
                random_state=random_state
            )
            status_text.text("Computing PaCMAP projection...")
            progress_bar.progress(30)
            reduced = reducer.fit_transform(embeddings)
            progress_bar.progress(100)

        elif method == "TriMAP":
            status_text.text("Initializing TriMAP...")
            progress_bar.progress(10)
            reducer = trimap.TRIMAP(n_dims=params["n_components"], n_inliers=params["n_neighbors"])
            status_text.text("Computing TriMAP projection...")
            progress_bar.progress(30)
            reduced = reducer.fit_transform(embeddings)
            progress_bar.progress(100)

        else:
            st.error(f"Unsupported dimensionality reduction method: {method}")
            return None

        status_text.text("Dimensionality reduction completed!")
        return reduced

    finally:
        def cleanup():
            import time
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
        
        cleanup()
