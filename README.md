To start chromadb:

```
docker-compose up -d --build
```

As a test dataset [`GoEmotions` dataset](https://research.google/blog/goemotions-a-dataset-for-fine-grained-emotion-classification/) by Google will be used. It is located in the data folder, and also you can read more about that on [GitHub](https://github.com/google-research/google-research/tree/master/goemotions). Once the Docker Compose with all services is set up, use the following command to upload the data to the database and use it:

```bash
python3 main.py
```

Execute `upload_goemotion_data()` function only once to upload all the data to the database, and once it's done, feel free to use `query_goemotion_data(str)` function to play around with it.
