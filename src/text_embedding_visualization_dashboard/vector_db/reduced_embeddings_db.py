import hashlib
import json
from typing import Dict, Optional
import numpy as np

from text_embedding_visualization_dashboard.vector_db import VectorDB


class ReducedEmbeddingsDB:
    def __init__(self, vector_db: VectorDB):
        self.vector_db = vector_db
        self.collection_name = "reduced_embeddings"
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        """Ensure the reduced embeddings collection exists."""
        self.vector_db.add_collection(self.collection_name)

    def _generate_cache_key(self, dataset_name: str, method: str, params: Dict) -> str:
        """Generate a unique key for the reduction configuration."""
        config_str = json.dumps({
            "dataset": dataset_name,
            "method": method,
            "params": params
        }, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()

    def store_reduced_embeddings(
        self, 
        dataset_name: str, 
        method: str, 
        params: Dict, 
        reduced_embeddings: np.ndarray
    ) -> str:
        """
        Store reduced embeddings in the database.
        
        Args:
            dataset_name: Name of the original dataset
            method: Dimensionality reduction method used
            params: Parameters used for reduction
            reduced_embeddings: The reduced embeddings array
            
        Returns:
            str: The cache key for the stored embeddings
        """
        cache_key = self._generate_cache_key(dataset_name, method, params)
        
        embeddings_list = reduced_embeddings.tolist()
        
        self.vector_db.add_items_to_collection(
            name=self.collection_name,
            texts=[f"reduced_{dataset_name}"],
            embeddings=embeddings_list,
            ids=[cache_key],
            metadata=[{
                "dataset_name": dataset_name,
                "method": method,
                "params": json.dumps(params),
                "dimensions": str(reduced_embeddings.shape[1])
            }]
        )
        
        return cache_key

    def get_reduced_embeddings(
        self, 
        dataset_name: str, 
        method: str, 
        params: Dict
    ) -> Optional[np.ndarray]:
        """
        Retrieve reduced embeddings from the database if they exist.
        
        Args:
            dataset_name: Name of the original dataset
            method: Dimensionality reduction method used
            params: Parameters used for reduction
            
        Returns:
            Optional[np.ndarray]: The reduced embeddings if found, None otherwise
        """
        cache_key = self._generate_cache_key(dataset_name, method, params)
        
        try:
            result = self.vector_db.query_collection_by_metadata(
                name=self.collection_name,
                metadata={"dataset_name": dataset_name, "method": method},
                include=["embeddings", "metadatas"]
            )
            
            if result["ids"] and result["ids"][0] == cache_key:
                stored_params = json.loads(result["metadatas"][0]["params"])
                if stored_params == params:
                    return np.array(result["embeddings"][0])
        except Exception:
            pass
            
        return None 