#!/usr/bin/env python

import os
import json
import argparse

import numpy as np
from transformers import AutoTokenizer
from scipy.special import softmax as sp_softmax
from collections import defaultdict

# from scipy.sparse.csr_matrix import argmax as sp_argmax

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute LLM classification accuracy from generated_text."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default='./_datasets/HPC/',
        help="Base data directory",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default='meta-llama/Llama-3.1-8B-Instruct',
        help="Model name / subdirectory ",
    )
    parser.add_argument(
        "--target_file",
        type=str,
        default='imdb_sms_interval_1_pairs_with_llama_activations.json',
        help="Target JSON file name",
    )
    parser.add_argument(
        "--label_feature",
        type=str,
        default='label_context_1',
        help="Name of the numerical label field to use (label_context_1 or label_context_2)",
    )
    parser.add_argument(
        "--task", 
        type=str, 
        default='sen_w_t1',
        help='Candidate tasks: sen_w_t1, sen_w_t2, sen_w_b, lay_w_t1, lay_w_t2, lay_w_b, and selective_attention'
        ) 
    return parser.parse_args()

def get_label_token_ids(tokenizer, label_feature: str):
    """
    Decide label words and mapping to numeric labels based on the label feature,
    then tokenize them (without special tokens) and return mapping:

        numeric_label (int) -> token_id (int)

    - For label_feature == 'label_context_1': 'spam' -> 1, 'legitimate' -> 0
    - Otherwise: 'positive' -> 1, 'negative' -> 0
    """
    if label_feature == "label_context_1":
        label_words = ("positive", "negative", "POSITIVE", "NEGATIVE")
        word_to_numeric = {"positive": 1, "negative": 0, "POSITIVE":1, "NEGATIVE": 0}
    else:
        label_words = ("spam", "legitimate", "SPAM", "LEGITIMATE")
        word_to_numeric = {"spam": 1, "legitimate": 0, "SPAM":1, "LEGITIMATE":0}

    numeric_to_token_id = defaultdict(set)

    print("=== Debug: label word → token IDs → numeric labels ===")
    for word in label_words:
        numeric_label = word_to_numeric[word]
        ids = tokenizer(word, add_special_tokens=False).input_ids
        if not ids:
            raise ValueError(f"No tokens produced for label word: {word}")
        first_id = ids[0]
        numeric_to_token_id[numeric_label].add(first_id)
        print(f"  word '{word}': tokens {ids} -> first_id={first_id} -> numeric_label={numeric_label}")
    print("======================================================")
    print(numeric_to_token_id)
    for key in numeric_to_token_id.keys():
        numeric_to_token_id[key] = list(numeric_to_token_id[key])
    return numeric_to_token_id

def main():
    args = parse_args()
    args.target_file = str(args.task) + "_" + args.target_file
    input_path = os.path.join(args.data_dir, args.model_name, args.task, args.target_file)
    print(f"Loading JSON data from: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} samples from JSON.")

    # Load tokenizer for label tokenization
    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # print(tokenizer.convert_ids_to_tokens([17914, 98227]))

    # Map numeric label (0/1) → token_id
    numeric_to_token_id = get_label_token_ids(tokenizer, args.label_feature)

    target_probs = []
    skipped = 0
    debug_print_limit = 10

    for idx, item in enumerate(data):
        # Get numeric ground-truth label
        if args.label_feature not in item:
            raise KeyError(f"Label feature '{args.label_feature}' not found in item index {idx}.")

        # labels may be float32 in JSON; convert robustly to int
        true_label_raw = item[args.label_feature]
        try:
            true_label = int(round(float(true_label_raw)))
        except Exception as e:
            print(f"[WARN] Could not cast label '{true_label_raw}' to int at sample {idx}: {e}")
            skipped += 1
            continue

        if true_label not in numeric_to_token_id:
            print(f"[WARN] True label {true_label} not in numeric_to_token_id at sample {idx}. Skipping.")
            skipped += 1
            continue

        target_token_id = numeric_to_token_id[true_label]

        logits_file = item.get("logits_new_file", None)
        if logits_file is None:
            print(f"[WARN] No 'logits_new_file' in item {idx}. Skipping.")
            skipped += 1
            continue

        # Use the path as it appears in JSON; if it's relative, it should be relative
        # to your current working directory when running this script.
        logits_path = logits_file

        if not os.path.isfile(logits_path):
            # Optional: try to join with project root or data_dir if you want.
            print(f"[WARN] Logits file not found at '{logits_path}' for sample {idx}. Skipping.")
            skipped += 1
            continue

        logits = np.load(logits_path)  # expected shape: [new_tokens, vocab_size]
        if logits.ndim != 2:
            print(f"[WARN] Unexpected logits shape {logits.shape} in sample {idx}. Expected 2D. Skipping.")
            skipped += 1
            continue

        # Take the first new token’s logits: shape [vocab_size]
        first_logits = logits[0]  # [vocab_size]

        # Convert to probabilities
        probs = sp_softmax(np.asarray(first_logits))
        # print('Sum probs=', sum(probs), probs.shape, len(probs), np.argmax(probs), max(probs))

        if any(id >= probs.shape[0] for id in target_token_id):
            print(
                f"[WARN] target_token_id {target_token_id} >= vocab_size {probs.shape[0]} "
                f"for sample {idx}. Skipping."
            )
            skipped += 1
            continue

        target_prob = float(probs[target_token_id[0]] + probs[target_token_id[1]])
        target_probs.append(target_prob)

        # Debug print for first few samples
        if idx < debug_print_limit:
            print(f"--- Sample {idx} ---")
            print(f"  true_label_raw: {true_label_raw}, true_label: {true_label}")
            print(f"  target_token_id: {target_token_id}")
            print(f"  logits_file: {logits_path}")
            print(f"  logits_shape: {logits.shape}")
            print(f"  first_logits[:10]: {first_logits[:10]}")
            print(f"  target_prob: {target_prob:.6f}")
            print("--------------------")

    if not target_probs:
        print("No valid probabilities collected. All samples were skipped.")
        return

    avg_prob = float(np.mean(target_probs))
    print("\n=== Summary ===")
    print(f"Number of valid samples: {len(target_probs)}")
    print(f"Skipped samples: {skipped}")
    print(f"Average probability of target tokens: {avg_prob:.6f}")


if __name__ == "__main__":
    main()
