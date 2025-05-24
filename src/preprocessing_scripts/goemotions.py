import pandas as pd


def preprocess_goemotions(dataset_id):
    df = pd.read_csv(f"../../data/goemotions_{dataset_id}.csv")

    df_processed = df[["text"]].copy()

    emotion_columns = df.columns[df.columns.get_loc("admiration") :]

    df_processed["label"] = df[emotion_columns].apply(
        lambda row: "-".join([emotion for emotion in emotion_columns if row[emotion] == 1]), axis=1
    )
    df_processed = df_processed[df_processed["label"] != ""]
    df_processed.to_csv(f"../../data/goemotions_processed_{dataset_id}.csv", index=False)


if __name__ == "__main__":
    preprocess_goemotions(1)
