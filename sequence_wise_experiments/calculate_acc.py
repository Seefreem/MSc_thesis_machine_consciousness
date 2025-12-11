#!/usr/bin/env python

import os
import json
import argparse

from transformers import AutoTokenizer

def get_label_token_mapping(tokenizer, label_feature: str):
    """
    Tokenize label words (without special tokens) and map their first token IDs to numerical labels.
    Default: "positive" -> 1, "negative" -> 0.
    """
    # # If you later want to special-case spam/legitimate, you can uncomment this block:
    # if label_feature == "label_context_1":
    #     label_words = ("spam", "legitimate")
    #     word_to_numeric = {"spam": 1, "legitimate": 0}
    # else:
    #     label_words = ("spam", "legitimate")
    #     word_to_numeric = {"spam": 1, "legitimate": 0}

    # token_id_to_label = {}
    # for w in label_words:
    #     ids = tokenizer(w, add_special_tokens=False).input_ids
    #     if not ids:
    #         raise ValueError(f"No tokens produced for label word: {w}")
    #     first_id = ids[0]
    #     token_id_to_label[first_id] = word_to_numeric[w]

    # print("=== Debug: label token mapping ===")
    # for word in label_words:
    #     ids = tokenizer(word, add_special_tokens=False).input_ids
    #     print(f"  word '{word}': tokens {ids} -> first_id={ids[0]} -> label={word_to_numeric[word]}")
    # print("==================================")
    # print(label_feature, token_id_to_label) 

    token_id_to_label = {}
    if label_feature == 'label_context_1': # sentiment classification
        token_id_to_label[tokenizer('positive', add_special_tokens=False).input_ids[0]] = 1
        token_id_to_label[tokenizer('negative', add_special_tokens=False).input_ids[0]] = 0
    elif label_feature == 'label_context_2': # Spam detection
        token_id_to_label[tokenizer('spam', add_special_tokens=False).input_ids[0]] = 1
        token_id_to_label[tokenizer('legitimate', add_special_tokens=False).input_ids[0]] = 0
    else:
        raise ValueError(f"Unknown label feature: {label_feature}")

    return token_id_to_label


def main(args):
    args.target_file = str(args.task) + "_" + args.target_file
    input_path = os.path.join(args.data_dir, args.model_name, args.task, args.target_file)
    print(f"Loading data from: {input_path}")

    # Load JSON file (assumed to be a list of dicts)
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} samples.")

    # Load tokenizer
    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Build mapping from label token IDs -> numerical labels
    token_id_to_label = get_label_token_mapping(tokenizer, args.label_feature)

    # Iterate data, map first token of generated_text to a label, compute accuracy
    correct = 0
    total = 0
    skipped = 0

    debug_print_limit = 10  # how many samples to print for debugging

    for idx, item in enumerate(data):
        true_label = item[args.label_feature]
        gen_text = item.get("generated_text", "").lower()

        # Tokenize generated_text without special tokens
        ids = tokenizer(gen_text, add_special_tokens=False).input_ids

        if not ids:
            skipped += 1
            if skipped <= 5:
                print(f"[WARN] Empty tokenization for sample {idx}, generated_text='{gen_text}'")
            continue

        first_id = ids[0]
        pred_label = token_id_to_label.get(first_id, None)

        if pred_label is None:
            # Unknown first token ID (not one of our label words)
            skipped += 1
            if skipped <= 5:
                print(
                    f"[WARN] First token id {first_id} for sample {idx} "
                    f"not found in label mapping. generated_text='{gen_text}'"
                )
            continue

        total += 1
        if pred_label == true_label:
            correct += 1

        # Debug print for first few samples
        if idx < debug_print_limit:
            print(f"--- Sample {idx} ---")
            print(f"  {args.label_feature} (true): {true_label}")
            print(f"  generated_text: {gen_text}")
            print(f"  token_ids: {ids}")
            print(f"  first_token_id: {first_id}")
            print(f"  predicted_label: {pred_label}")
            print("--------------------")

    if total == 0:
        print("No valid predictions (total=0). Possibly all predictions were skipped.")
        return

    accuracy = correct / total
    print("\n=== Results ===")
    print(f"Correct predictions: {correct}")
    print(f"Total counted predictions: {total}")
    print(f"Skipped samples (empty or unmapped first token): {skipped}")
    print(f"Overall accuracy: {accuracy:.4f}")


if __name__ == "__main__":
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
        default='imdb_sms_interval_1_pairs_with_activations.json',
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
    args = parser.parse_args()
    main(args)
