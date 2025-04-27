from tqdm import tqdm
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from src.text_embedding_visualization_dashboard.vector_db.db import VectorDB

class Embeddings:
    def __init__(self, vector_db: VectorDB, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.vector_db = vector_db

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        return self.model.encode(text).tolist()
    
    def batch_process_texts(
        self,
        texts: List[str],
        collection_name: str,
        metadatas: Optional[List[dict]] = None,
        batch_size: int = 128
    ):
        """
        Batch process texts to generate embeddings and store them in ChromaDB.
        
        :param texts: List of texts to embed.
        :param collection_name: Name of the collection in ChromaDB.
        :param metadatas: Optional list of metadata dicts for each text.
        :param batch_size: Number of samples per batch.
        """
        texts = [text for text in texts if text and text.strip()]
        if not texts: raise ValueError("No valid texts provided for batch processing.")

        self.vector_db.add_collection(name=collection_name)

        num_batches = (len(texts) + batch_size - 1) // batch_size

        for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches", total=num_batches):
            batch_texts = texts[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size] if metadatas else None

            embeddings = self.generate_embedding(batch_texts)
            ids = [f"doc_{j}" for j in range(i, i + len(batch_texts))]

            if batch_metadatas is None:
                batch_metadatas = [{"source": "batch_process"} for _ in batch_texts]

            self.vector_db.add_items_to_collection(
                name=collection_name,
                texts=batch_texts,
                embeddings=embeddings,
                ids=ids,
                metadata=batch_metadatas,
            )
    
    def query_similar_texts(
        self,
        query_text: str,
        collection_name: str,
        top_k: int = 5
    ):
        """
        Query similar texts from a collection in ChromaDB.

        :param query_text: Text to search for similar documents.
        :param collection_name: Name of the collection to search in.
        :param top_k: Number of most similar results to return.
        :return: List of results with id, text, metadata and distance.
        """
        query_embedding = self.generate_embedding(query_text)

        collection = self.vector_db.get_collection(name=collection_name)

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        formatted_results = []
        for i in range(len(results["ids"][0])):
            formatted_results.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })

        return formatted_results
    
    def batch_query_similar_texts(
        self,
        query_texts: List[str],
        collection_name: str,
        top_k: int = 5,
    ) -> List[List[dict]]:
        """
        Batch query similar texts for a list of input queries.

        :param query_texts: List of query texts.
        :param collection_name: Collection to search in.
        :param top_k: Number of top results per query.
        :return: List of results for each query.
        """
        query_embeddings = self.generate_embedding(query_texts)

        collection = self.vector_db.get_collection(name=collection_name)

        results = collection.query(
            query_embeddings=query_embeddings,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        batch_results = []
        for query_idx in range(len(query_texts)):
            single_query_results = []
            for i in range(len(results["ids"][query_idx])):
                single_query_results.append({
                    "id": results["ids"][query_idx][i],
                    "text": results["documents"][query_idx][i],
                    "metadata": results["metadatas"][query_idx][i],
                    "distance": results["distances"][query_idx][i],
                })
            batch_results.append(single_query_results)

        return batch_results
