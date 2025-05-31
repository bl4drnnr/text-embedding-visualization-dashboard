<h1 align="center">Text Embeddubgs Visualization Dashboard</h1>

Due to the fact that this project is using **uv** as a package manager, there is a little different sequence of commands that you need to use in order to set it up locally:

```bash
uv init text-embedding-visualization-dashboard --python 3.12
rm -rf text-embedding-visualization-dashboard
uv venv .venv
source .venv/bin/activate
uv sync
```

To start the application, run:

```bash
# Just database
docker-compose --profile database up -d 

# Entire app
docker compose --profile full-app up -d 
```

> **WARNING**: The approach that has been shown above is going to create a container with the size of 20-25GB. In the case you want to have application running in the more optimal way, you can execute it in the following way:

```bash
docker-compose up server
streamlit run src/text_embedding_visualization_dashboard/frontend/frontend.py
```

After the containers are up, open your browser and navigate to: `http://localhost:8501/`

As a test dataset [`GoEmotions` dataset](https://research.google/blog/goemotions-a-dataset-for-fine-grained-emotion-classification/) by Google will be used. It is located in the data folder, and also you can read more about that on [GitHub](https://github.com/google-research/google-research/tree/master/goemotions). Once the Docker Compose with all services is set up, use the following command to upload the data to the database and use it:


```bash
python3 main.py --mode upload --model all-MiniLM-L6-v2 --dataset data/goemotions_processed_1.csv data/goemotions_processed_2.csv data/goemotions_processed_3.csv --collection goemotions_embeddings
```

Also you can get more information about how to use `main.py` by simply typing `python3 main.py --help`. CLI is avaialbe for this project and you can not only upload the test dataset to see how an application works, but also test the query that is available via CLI.

```bash
python3 main.py --help

usage: main.py [-h] --mode {upload,query}
               [--model MODEL]
               [--dataset DATASET [DATASET ...]]
               [--collection COLLECTION]
               [--batch-size BATCH_SIZE]
               [--query QUERY]
               [--top-k TOP_K]

Text Embedding Generator for Datasets and VectorDB Storage

options:
  -h, --help            show this help message and exit
  --mode {upload,query}
                        Mode: 'upload' to upload data, 'query' to search similar texts.
  --model MODEL         Name of the SentenceTransformer model to use.
  --dataset DATASET [DATASET ...]
                        Paths to one or more CSV files containing text data.
  --collection COLLECTION
                        Name of the collection to store embeddings in ChromaDB.
  --batch-size BATCH_SIZE
                        Batch size for embedding generation.
  --query QUERY         Query text to search for similar documents (required for query mode).
  --top-k TOP_K         Number of top similar results to return for a query (default: 5).

        Example usage:

        python3 main.py --mode upload --model all-MiniLM-L6-v2 --dataset data/goemotions_1.csv data/goemotions_2.csv --collection goemotions_embeddings

        python3 main.py --mode query --model all-MiniLM-L6-v2 --collection goemotions_embeddings --query "I love sunny days" --top-k 5
```
