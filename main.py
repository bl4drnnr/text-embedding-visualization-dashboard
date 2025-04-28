import pandas as pd
from src.text_embedding_visualization_dashboard.embeddings import Embeddings
from src.text_embedding_visualization_dashboard.vector_db.db import VectorDB
from src.text_embedding_visualization_dashboard.utils.parse_args import parse_args


# UWAGA: Tu w tym momencie są tylko przykładowe testowe funkcje, aby zobaczyć, że ta baza oraz embeddingi działają.
def main():
    args = parse_args()

    model_name = args.model
    dataset = args.dataset
    collection = args.collection
    batch_size = args.batch_size
    mode = args.mode
    query_text = args.query
    top_k = args.top_k

    vector_db = VectorDB()
    service = Embeddings(vector_db, model_name)

    def upload_goemotion_data():
        dfs = [pd.read_csv(file) for file in dataset]
        df = pd.concat(dfs, ignore_index=True)

        texts = df["text"].tolist()

        if "labels" in df.columns:
            metadatas = [{"label": str(label)} for label in df["labels"]]
        else:
            metadatas = None

        service.batch_process_texts(texts=texts, collection_name=collection, metadatas=metadatas, batch_size=batch_size)

    def query_goemotion_data(query: str):
        results = service.query_similar_texts(query_text=query, collection_name=collection, top_k=top_k)

        print(f'\nWyniki dla zapytania: "{query}":\n')
        for idx, item in enumerate(results):
            print(f"{idx + 1}. {item['text']} (Distance: {item['distance']:.4f}) Metadata: {item['metadata']}")

    if mode == "upload":
        if not dataset:
            raise ValueError("Musisz podać plik lub pliki CSV (--dataset), żeby załadować dane.")
        upload_goemotion_data()

    elif mode == "query":
        if not query_text:
            raise ValueError("Musisz podać tekst zapytania (--query), żeby wyszukać podobne teksty.")
        query_goemotion_data(query_text)


if __name__ == "__main__":
    main()
