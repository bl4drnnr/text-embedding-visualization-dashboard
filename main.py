import pandas as pd
from src.text_embedding_visualization_dashboard.embeddings import Embeddings
from src.text_embedding_visualization_dashboard.vector_db.db import VectorDB

# UWAGA: Tu w tym momencie są tylko przykładowe testowe funkcje, aby zobaczyć, że ta baza oraz embeddingi działają.
def main():
    vector_db = VectorDB()
    service = Embeddings(vector_db)

    def upload_goemotion_data():
        data_files = ["data/goemotions_1.csv", "data/goemotions_2.csv", "data/goemotions_3.csv"]
        dfs = [pd.read_csv(file) for file in data_files]
        df = pd.concat(dfs, ignore_index=True)
        
        texts = df["text"].tolist()

        if "labels" in df.columns:
            metadatas = [{"label": str(label)} for label in df["labels"]]
        else:
            metadatas = None

        service.batch_process_texts(
            texts=texts,
            collection_name="goemotions_embeddings",
            metadatas=metadatas,
            batch_size=128
        )

    def query_goemotion_data(query: str):
        results = service.query_similar_texts(query_text=query, collection_name="goemotions_embeddings", top_k=5)

        print(f"\nWyniki dla zapytania: \"{query}\":\n")
        for idx, item in enumerate(results):
            print(f"{idx+1}. {item['text']} (Distance: {item['distance']:.4f}) Metadata: {item['metadata']}")

    # Zauploadujcie sobie dane jeden raz i można to zakomentować
    upload_goemotion_data()
    # Tutaj możecie sobie podstawić jakieś query i pobawić się tym
    query_goemotion_data("I like Canada.")


if __name__ == "__main__":
    main()
