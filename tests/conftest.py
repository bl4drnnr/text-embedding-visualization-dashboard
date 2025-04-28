import pytest
from text_embedding_visualization_dashboard.vector_db import VectorDB
from sentence_transformers import SentenceTransformer


@pytest.fixture(scope="session")
def vector_db():
    """Provides a VectorDB instance for testing."""
    db = VectorDB()  # adjust if it needs config
    yield db
    # If there's a close/cleanup method
    if hasattr(db, "close"):
        db.close()


@pytest.fixture(scope="session")
def collection_name():
    return "test-collection"


@pytest.fixture(scope="session")
def embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")
