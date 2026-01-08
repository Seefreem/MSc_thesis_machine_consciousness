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
    json_path: str = os.path.join("_datasets/hahackathon_subsets", "combined_subset.json"),
):
    """
    Load combined_subset.json and convert it into a tokenized HuggingFace Dataset
    suitable for Transformer models.

    - Input text:  text_column (default: 'text')
    - Labels:      is_humor (binary), is_offensive (binary), ...
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    ds = Dataset.from_list(data).cast_column("is_humor", Value("float32"))
    ds = ds.cast_column("is_offensive", Value("float32"))

    return data, ds

# def main():
#     imdb_sms = load_imdb_sms_for_transformer()
#     print(imdb_sms.select([1]).features)
#     print(imdb_sms.select([1]).to_dict())

#     hahackathon_combined = load_hahackathon_combined_for_transformer()
#     print(hahackathon_combined.select([1]).features)
#     print(hahackathon_combined.select([1]).to_dict())
    

# if __name__ == "__main__":
#     main()
