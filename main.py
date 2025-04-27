from src.text_embedding_visualization_dashboard.embeddings import Embeddings
from src.text_embedding_visualization_dashboard.vector_db.db import VectorDB

# UWAGA: Tu w tym momencie są tylko przykładowe testowe funkcje, aby zobaczyć, że ta baza oraz embeddingi działają.
def main():
    vector_db = VectorDB()
    service = Embeddings(vector_db)

    texts = ["I love pizza!", "The sun is shining.", "I hate rainy days."]
    service.batch_process_texts(texts=texts, collection_name="test_embeddings")

    query = "I really enjoy eating pizza."
    results = service.query_similar_texts(query_text=query, collection_name="test_embeddings")

    for idx, item in enumerate(results):
        print(f"{idx+1}. {item['text']} (Distance: {item['distance']:.4f})")


if __name__ == "__main__":
    main()
