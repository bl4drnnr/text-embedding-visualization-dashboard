import pytest
from text_embedding_visualization_dashboard.vector_db import VectorDB


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
