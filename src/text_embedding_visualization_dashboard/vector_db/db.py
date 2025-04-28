import chromadb
from text_embedding_visualization_dashboard.utils import setup_logger
from text_embedding_visualization_dashboard.models.vectordb_models import QUERY_INCLUDE, GET_INCLUDE
from datetime import datetime
from typing import Literal, Sequence
from chromadb.api import Collection, QueryResult, GetResult
import os


logger = setup_logger("chroma_db-logger")

chroma_host = os.getenv("CHROMA_HOST", "localhost")
chroma_port = int(os.getenv("CHROMA_PORT", "8800"))


class VectorDB:
    def __init__(self):
        self.client = chromadb.HttpClient(
            host=chroma_host,
            port=chroma_port,
        )

    def get_all_collections(self) -> Sequence[Collection]:
        """
        Get all collections from ChromaDb
        :return:
        """
        logger.info("Getting all collections")
        return self.client.list_collections()

    def get_collection(self, name: str) -> Collection:
        """
        Get collection by name from ChromaDb
        :param name:
        :return: Collection name
        """
        logger.info(f"Getting collection {name}")
        return self.client.get_collection(name)

    def add_collection(self, name: str, distance: Literal["cosine", "l2", "ip"] = "cosine") -> None:
        """
        Adds a collection to the ChromaDb
        :param name: name of the collection
        :param distance: Metric for calculating the distance between two vectors: cosine | L2 | inner product
        :return:
        """
        if name not in [elem.name for elem in self.get_all_collections()]:
            self.client.create_collection(
                name=name,
                metadata={"created": str(datetime.now())},
                configuration={
                    "hnsw": {"space": distance},
                    # TODO: Replace it with our embedding function
                    # "embedding_function": cohere_ef
                },
            )
            logger.info(f"Created collection {name}")
        else:
            logger.info(f"Collection {name} already exists.")

    def delete_collection(self, name: str) -> None:
        """
        Deletes a collection
        :param name: ChromaDb collection name
        :return: None
        """
        logger.info(f"Deleting collection {name}")
        if name not in [elem.name for elem in self.get_all_collections()]:
            logger.error(f"Collection {name} does not exist.")

        else:
            self.client.delete_collection(name)

    def add_items_to_collection(
        self,
        name: str,
        texts: list[str],
        embeddings: list[list[float]],
        ids: list[str] | None = None,
        metadata: list[dict[str, str]] | None = None,
    ) -> None:
        """ "
        Adds items to a collection
        :param name: collection name
        :param texts: Texts to store in db
        :param embeddings: Texts embeddings
        :param ids: Ids to store in db, None by default
        :param metadata: Metadata to store in db, None by default. Eg: {{"label": "ClassA"}, {"label": "ClassB"}}
        :return:None"""

        collection = self.get_collection(name)
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(texts))]

        add_kwargs = {
            "documents": texts,
            "embeddings": embeddings,
            "ids": ids,
        }

        if metadata is not None:
            add_kwargs["metadatas"] = metadata

        collection.add(**add_kwargs)
        logger.info(f"Added {len(texts)} items to {collection.name}")

    def query_collection(
        self,
        name: str,
        query_embeddings: list[float],
        n_results: int = 5,
        include: QUERY_INCLUDE = [
            "metadatas",
            "documents",
            "distances",
        ],
    ) -> QueryResult:
        """
        Query collection
        :param name: collection name
        :param query: Text embedding to search for
        :param n_results: Results to return
        :param include: A list of what to include in the results. Can contain `"embeddings"`, `"metadatas"`, `"documents"`, `"distances"`. Ids are always included. Defaults to `["metadatas", "documents", "distances"]`. Optional.

        :return:"""
        collection = self.get_collection(name)
        # TODO: Here we should use our own function for embedding, or pass that function to chroma client during creation of collection:
        # https://cookbook.chromadb.dev/embeddings/bring-your-own-embeddings/
        results = collection.query(
            query_embeddings=query_embeddings,  # Chroma will embed this for you
            n_results=n_results,  # how many results to return,
            include=include,
        )

        return results

    def query_collection_by_metadata(
        self, name: str, metadata: dict, include: GET_INCLUDE = ["documents", "metadatas"]
    ) -> GetResult:
        """
        Query collection by metadata. Useful for grabbing all items with same label.
        :param name: collection name
        :param metadata: Metadata to search for, eg: {"label" : "ClassA"}
        :param include: A list of what to include in the results. Can contain `"embeddings"`, `"metadatas"`, `"documents"``. Ids are always included. Defaults to `["metadatas", "documents"]`. Optional.

        :return:
        """
        logger.info(f"Querying collection {name} by metadata {metadata}")
        collection = self.get_collection(name)
        return collection.get(where=metadata, include=include)

    def get_all_items_from_collection(
        self,
        name: str,
        include: GET_INCLUDE = [
            "metadatas",
            "documents",
        ],
    ) -> GetResult:
        """
        Get all items from collection
        :param name: collection name
        :param include: A list of what to include in the results. Can contain `"embeddings"`, `"metadatas"`, `"documents"``. Ids are always included. Defaults to `["metadatas", "documents"]`. Optional.

        :return:
        """
        logger.info(f"Getting all items from collection {name}")
        collection = self.get_collection(name)
        return collection.get(include=include)
