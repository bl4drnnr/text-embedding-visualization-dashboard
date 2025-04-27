from typing import Dict, Tuple

import streamlit as st
import pandas as pd
import umap
import trimap
import pacmap
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer
import numpy as np

from text_embedding_visualization_dashboard.vector_db import VectorDB


def create_embeddings(db: VectorDB, uploaded_file) -> str | None:
    """
    Creates embeddings from uploaded CSV file and adds them to the vector database.

    Parameters:
    db : VectorDB
        The vector database instance for storing embeddings.
    uploaded_file : file object
        CSV file uploaded by the user. Must contain 'text' and 'label' columns.

    Returns:
    str or None
        The name of the created collection, or None if there was an error.
    """

    df = pd.read_csv(uploaded_file)

    if "text" not in df.columns:
        st.error("The CSV file must contain a 'text' column.")
        return None

    if "label" not in df.columns:
        st.error("The CSV file must contain a 'label' column.")
        return None

    texts = df["text"].tolist()
    labels = df["label"].tolist()

    collection_name = uploaded_file.name[:-4]
    db.add_collection(collection_name)

    # TODO:
    # Just an example remove later
    ids = [f"doc_{i}" for i in range(len(texts))]
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts).tolist()
    metadatas = [{"label": label} for label in labels]

    db.add_items_to_collection(collection_name, texts, embeddings, ids, metadatas)

    return collection_name


def get_embeddings(db: VectorDB, dataset_name: str) -> Tuple[list[list[float]], list[str]]:
    """
    Returns embeddings and corresponding labels from the database.

    Parameters:
    db : VectorDB
        The vector database instance for retrieving embeddings.
    dataset_name : str
        Name of the collection to retrieve embeddings from.

    Returns:
    tuple
        (embeddings, labels): The retrieved embeddings and corresponding labels.
    """

    db_collection = db.get_all_items_from_collection(dataset_name, include=["embeddings", "metadatas"])

    embeddings = db_collection["embeddings"]

    metadatas = db_collection["metadatas"]

    labels = [matadata.get("label") for matadata in metadatas]

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

    if method == "UMAP":
        reducer = umap.UMAP(
            n_neighbors=params["n_neighbors"],
            min_dist=params["min_dist"],
            n_components=params["n_components"],
            random_state=random_state,
        )
        reduced = reducer.fit_transform(embeddings)

    elif method == "t-SNE":
        reducer = TSNE(
            n_components=params["n_components"],
            perplexity=params["perplexity"],
            max_iter=params["max_iter"],
            random_state=random_state,
        )
        reduced = reducer.fit_transform(embeddings)

    elif method == "PaCMAP":
        reducer = pacmap.PaCMAP(
            n_neighbors=params["n_neighbors"], n_components=params["n_components"], random_state=random_state
        )
        reduced = reducer.fit_transform(embeddings)

    elif method == "TriMAP":
        reducer = trimap.TRIMAP(n_dims=params["n_components"], n_inliers=params["n_neighbors"])
        reduced = reducer.fit_transform(embeddings)

    else:
        st.error(f"Unsupported dimensionality reduction method: {method}")

    return reduced