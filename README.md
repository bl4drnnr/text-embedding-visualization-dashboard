To start chromadb:

```
docker-compose up -d --build
```

In order to upload the test data ([`GoEmotions` dataset](https://research.google/blog/goemotions-a-dataset-for-fine-grained-emotion-classification/) in the data folder, [also on GH](https://github.com/google-research/google-research/tree/master/goemotions)), once the database is set up:

```bash
python3 main.py
```

Execute `upload_goemotion_data()` function only once to upload all the data to the database, and once it's done, feel free to use `query_goemotion_data(str)` function to play around with it.
