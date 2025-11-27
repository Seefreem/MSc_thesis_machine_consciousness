import json
import os

import pandas as pd
from datasets import Dataset, Features, Value


def load_imdb_sms_for_transformer(
    json_path: str = "_datasets.filtered_data/imdb_sms_interval_1_pairs.json",
):
    """
    Load imdb_sms_interval_1_pairs.json and convert it into a
    HuggingFace Dataset suitable for Transformer models.

    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ds = Dataset.from_list(data).cast_column("label_context_1", Value("float32"))
    ds = ds.cast_column("label_context_2", Value("float32"))

    return data, ds


def load_hahackathon_combined_for_transformer(
    csv_path: str = os.path.join("_datasets.hahackathon_subsets", "combined_subset.csv"),
):
    """
    Load combined_subset.csv and convert it into a tokenized HuggingFace Dataset
    suitable for Transformer models.

    - Input text:  text_column (default: 'text')
    - Labels:      is_humor (binary), humor_rating (float), offense_rating (float)
    """
    df = pd.read_csv(csv_path)

    ds = Dataset.from_pandas(df)

    # Ensure label dtypes (optional but often useful)
    if "is_humor" in ds.column_names:
        # ds = ds.cast_column("is_humor", ds.features["is_humor"].cast("int64"))
        ds = ds.cast_column("is_humor", Value("float32"))
    if "is_offensive" in ds.column_names:
        # ds = ds.cast_column("is_offensive", ds.features["is_offensive"].cast("int64"))
        ds = ds.cast_column("is_offensive", Value("float32"))
    if "humor_rating" in ds.column_names:
        # ds = ds.cast_column("humor_rating", ds.features["humor_rating"].cast("float32"))
        ds = ds.cast_column("humor_rating", Value("float32"))
    if "offense_rating" in ds.column_names:
        # ds = ds.cast_column("offense_rating", ds.features["offense_rating"].cast("float32"))
        ds = ds.cast_column("offense_rating", Value("float32"))

    return ds

# def main():
#     imdb_sms = load_imdb_sms_for_transformer()
#     print(imdb_sms.select([1]).features)
#     print(imdb_sms.select([1]).to_dict())

#     hahackathon_combined = load_hahackathon_combined_for_transformer()
#     print(hahackathon_combined.select([1]).features)
#     print(hahackathon_combined.select([1]).to_dict())
    

# if __name__ == "__main__":
#     main()
