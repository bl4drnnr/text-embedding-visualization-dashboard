import chromadb
from text_embedding_visualization_dashboard.utils import cfg, setup_logger
from datetime import datetime
from chromadb.api.models.Collection import Collection
from chromadb.errors import ChromaError

logger = setup_logger("chroma_db-logger")

class VectorDB:
    def __init__(self):
        logger.info(cfg.VECTOR_DB_HTTP_PORT)
        self.client = chromadb.HttpClient(
            host="localhost",
            port=cfg.VECTOR_DB_HTTP_PORT,
        )



    def get_all_collections(self):
        logger.info("Getting all collections")
        return self.client.list_collections()

    def get_collection(self, name: str):
        logger.info(f"Getting collection {name}")
        return self.client.get_collection(name)


    def add_collection(self, name: str):
        logger.info(self.get_all_collections())
        if name not in [elem.name for elem in self.get_all_collections()]:
            self.client.create_collection(name = name, metadata={"created": str(datetime.now())})
            logger.info(f"Created collection {name}")

        else:
            logger.info(f"Collection {name} already exists.")

    def delete_collection(self, name: str):
        logger.info(f"Deleting collection {name}")
        if name not in [elem.name for elem in self.get_all_collections()]:
            logger.error(f"Collection {name} does not exist.")

        else:
            self.client.delete_collection(name)

    @staticmethod
    def add_items_to_collection(collection: Collection, texts: list[str], embeddings: list[list[float]], ids: list[str] | None = None):

        if ids is None:
            ids = [f"doc_{i}" for i in range(len(texts))]

        collection.add(
            documents=texts,
            embeddings=embeddings,
            ids=ids
        )
        logger.info(f"Added {len(texts)} items to {collection.name}")

    @staticmethod
    def query_collection(collection: Collection, query: str, n_results: int = 5, include=['documents', 'distances', 'metadatas']):
        #TODO: Here we should use our own function for embedding, or pass that function to chroma client during creation of collection:
        #https://cookbook.chromadb.dev/embeddings/bring-your-own-embeddings/
        results = collection.query(
            query_texts=[query],  # Chroma will embed this for you
            n_results=n_results,  # how many results to return,
            include=include
        )

        return results