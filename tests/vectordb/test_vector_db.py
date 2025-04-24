from sentence_transformers import SentenceTransformer

def test_add_collection(vector_db, collection_name):
    collections_number_before_adding = len(vector_db.get_all_collections())
    vector_db.add_collection(collection_name)
    all_collections = vector_db.get_all_collections()
    collections_number_after_adding = len(all_collections)
    assert collection_name in [elem.name for elem in all_collections]
    assert collections_number_before_adding == collections_number_after_adding - 1

def test_add_documents(vector_db, collection_name):
    texts = [
        # 🍎 Fruits
        "Apples are red",
        "I like eating bananas",
        "Oranges are full of vitamin C",

        # 🏀 Sports
        "Soccer is a popular sport",
        "Messi scored a goal",
        "Gym training increases strength",

        # 💻 Technology
        "Python is a programming language",
        "Computers process data",
        "Artificial intelligence is changing the world"
    ]

    ids = [f"doc_{i}" for i in range(len(texts))]
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts).tolist()
    collection = vector_db.get_collection(collection_name)
    vector_db.add_items_to_collection(collection, texts,embeddings, ids)
    documents_data = collection.get()

    assert  documents_data.get("documents")


def test_query_collection(vector_db, collection_name):
    collection = vector_db.get_collection(collection_name)
    results = vector_db.query_collection(collection, query="Do you know something about sport?", n_results=3)
    documents = results.get("documents")[0]

    assert len(documents) == 3
    assert "Soccer is a popular sport" in documents
    assert "Gym training increases strength" in documents


def test_delete_collection(vector_db, collection_name):
    collections_number_before_deleting = len(vector_db.get_all_collections())
    vector_db.delete_collection(collection_name)
    collections_number_after_deleting = len(vector_db.get_all_collections())

    assert collections_number_before_deleting == collections_number_after_deleting + 1

