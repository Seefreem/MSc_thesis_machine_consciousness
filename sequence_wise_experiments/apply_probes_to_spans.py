import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity  # not strictly needed here
import joblib
from transformers import AutoTokenizer
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

from myutilities import set_seed, load_metadata, infer_dims
from myutilities import compute_token_spans_for_sample
from myutilities import compute_token_spans_for_sample_task_as_pfx
from myutilities import compute_token_spans_for_exp2_samples
from myutilities import compute_token_spans_for_exp2_samples_task_as_pfx
from myutilities import save_updated_json
from myutilities import load_probes
from myutilities import compute_avg_acc_mag
from myutilities import save_matrices
from myutilities import plot_acc 
from myutilities import plot_magnitude
from myutilities import SPAN_LABELS, SPAN_LABELS_EXP2, SPAN_LABELS_TAP, SPAN_LABELS_EXP2_TAP

def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply logistic regression probes to span-averaged representations."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./_datasets/HPC/",
        help="Base data directory",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model name / subdirectory",
    )
    parser.add_argument(
        "--target_file",
        type=str,
        default="data_with_activations.json",
        help="Target JSON file name",
    )
    parser.add_argument(
        "--label_feature",
        type=str,
        default="label_context_1",
        help="Name of the numerical label field to use (label_context_1 or label_context_2)",
    )
    parser.add_argument(
        "--layer_idx",
        type=str,
        default="all",
        help='Layer index (e.g. "0", "5") or "all" to apply probes on all layers',
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=45,
        help="Random seed (for reproducibility if needed)",
    )
    parser.add_argument(
        "--task", 
        type=str, 
        default='sen_w_t1',
        help=('Candidate tasks: sen_w_t1, sen_w_t2, sen_w_b, '
              'sen_w_t1_swp_cnt, sen_w_t2_swp_cnt, sen_w_b_swp_cnt, '
              'sen_w_t1_task_as_pfx, sen_w_t2_task_as_pfx, sen_w_b_task_as_pfx, '
              'lay_w_t1_task_as_pfx, lay_w_t2_task_as_pfx, lay_w_b_task_as_pfx, '
              'lay_w_t1, lay_w_t2, lay_w_b, and selective_attention')
        ) 
    parser.add_argument(
        "--probe_base_dir", 
        type=str, 
        default="_datasets",
        ) 
    parser.add_argument(
        "--probe_type", 
        type=str, 
        default='normalized',
        help='inf: un-normalized, for activation intervention; normalized: normalized, for information probing'
    ) 
    parser.add_argument(
        "--intervened_layer_idx",
        type=int,
        default=None,
        help='The layer that is modified (e.g. "0", "5")',
    )
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)

    # base_dir = os.path.join(args.data_dir, args.model_name, args.task)
    if args.intervened_layer_idx != None:
        base_dir = os.path.join(args.data_dir, args.model_name, 
            args.task, str(args.intervened_layer_idx))
    else:
        base_dir = os.path.join(args.data_dir, args.model_name, 
            args.task)
    target_file = str(args.task) + "_" + args.target_file
    json_path = os.path.join(base_dir, target_file)

    print(f"Loading metadata from {json_path}")
    data = load_metadata(json_path)
    n_samples = len(data)
    print(f"Loaded {n_samples} samples.")

    # Determine layers
    first_hidden_path = data[0]["hidden_states_file"]
    _, n_layers_total, feat_dim = infer_dims(first_hidden_path)
    print(f"Detected n_layers={n_layers_total}, feat_dim={feat_dim}")

    if args.layer_idx.lower() == "all":
        layer_indices = list(range(n_layers_total))
    else:
        layer_indices = [int(args.layer_idx)]

    print(f"Will use probes for layers: {layer_indices}")

    # Load tokenizer
    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    # ---- Step 5: compute token id spans and update JSON ----
    print("Computing token spans per sample...")
    all_token_spans = []
    span_labels = SPAN_LABELS
    for i, sample in enumerate(data):
        if 'sen_w' in args.task:
            if 'task_as_pfx' in args.task:
                token_span_ranges, input_ids = compute_token_spans_for_sample_task_as_pfx(
                    args.model_name, sample, tokenizer, args.task
                )
                span_labels = SPAN_LABELS_TAP
            else:
                token_span_ranges, input_ids = compute_token_spans_for_sample(
                    args.model_name, sample, tokenizer, args.task
                )
                span_labels = SPAN_LABELS
        elif 'lay_w' in args.task:
            if 'task_as_pfx' in args.task:
                token_span_ranges, input_ids = compute_token_spans_for_exp2_samples_task_as_pfx(
                    args.model_name, sample, tokenizer, args.task
                )
                span_labels = SPAN_LABELS_EXP2_TAP
            else:
                token_span_ranges, input_ids = compute_token_spans_for_exp2_samples(
                    args.model_name, sample, tokenizer, args.task
                )
                span_labels = SPAN_LABELS_EXP2
        else:
            raise ValueError(f'Unhandled task type {args.task}')
        sample["token_spans"] = token_span_ranges
        # Optionally also store tokenized length
        sample["prompt_num_tokens"] = len(input_ids)
        all_token_spans.append(token_span_ranges)
        if (i + 1) % 100 == 0 or i == n_samples - 1:
            print(f"  Processed {i + 1}/{n_samples} samples for token spans")

    # Save updated JSON with token spans
    json_with_spans_path = os.path.join(
        base_dir,
        args.target_file.replace(".json", "_with_spans.json"),
    )
    save_updated_json(data, json_with_spans_path)

    # ---- Step 6 & 7: average pooling per span, apply probes ----
    # Prepare global containers for accuracy + projection magnitude
    n_spans = len(span_labels)
    n_layers = len(layer_indices)

    layer_results = {}
    all_probes = {}
    output_dub_dir = ''
    parent_dir = ''
    if args.probe_type == 'normalized':
        parent_dir = "linear_probes_logreg_normalized"
        output_dub_dir = "span_probe_results_logreg_normalized"
    else:
        parent_dir = "linear_probes_logreg_inf"
        output_dub_dir = "span_probe_results_logreg_inf"
    # Load all the needed probes
    all_probes = load_probes(args, layer_indices, parent_dir, args.task)
    for l_idx in layer_indices:
        layer_results[l_idx] = {}
        for span_label in span_labels:
            layer_results[l_idx][span_label] = {
                "y_true": [],
                "y_pred": [],
                "proj": [],
            }

    # process all the samples
    
    for samp_idx, sample in enumerate(data):
        # y_true[i] = int(sample[args.label_feature])
        hidden_file = sample["hidden_states_file"]
        hidden_path = hidden_file

        hs = np.load(hidden_path, mmap_mode="r")  # [all_tokens, n_layers, feat_dim]
        all_tokens = hs.shape[0]
        # print('np.load', hidden_path)

        token_spans = sample["token_spans"]
        prompt_num_tokens = sample.get("prompt_num_tokens", all_tokens)
        prompt_num_tokens = min(prompt_num_tokens, all_tokens)
        for li, lay_idx in enumerate(layer_indices):
            # layer_results[lay_idx]['y_true'][samp_idx] = int(sample[args.label_feature])
            # print('Probes from layer', lay_idx)
            scaler = all_probes[lay_idx]["scaler"]
            clf = all_probes[lay_idx]["classifier"]
            w = clf.coef_.reshape(-1)
            w_norm = np.linalg.norm(w) + 1e-9
            w_unit = w / w_norm
            for s_idx, span_label in enumerate(span_labels):
                start, end = token_spans[span_label]
                # Clip into prompt token range, ignore generated tokens
                start_clipped = max(0, min(start, prompt_num_tokens))
                end_clipped = max(0, min(end, prompt_num_tokens))

                # Average pooling across tokens of this span
                feat = hs[start_clipped:end_clipped, lay_idx, :]  # [n_span_tokens, feat_dim]
                # feat = span_reps.mean(axis=0)  # [feat_dim]
                # for debugging
                if samp_idx < 2 and li == 0:
                    print('hs.shape:', hs.shape)
                    print(f'Check: label: {span_label}; range{token_spans[span_label]}')
                    tokens_in_span = tokenizer([sample['wrapped_prompt']], 
                        return_tensors="pt", padding=False, 
                        truncation=True)['input_ids'][0][start_clipped:end_clipped]
                    print(f"tokens in this span: {tokenizer.convert_ids_to_tokens(tokens_in_span)}")

                # Apply probe: scale -> predict -> projection
                try:
                    check_is_fitted(scaler)
                    feat_scaled = scaler.transform(feat)  # [BS==1, feat_dim]
                except NotFittedError as exc:
                    # print(f"Note scaler is not fitted yet.")
                    feat_scaled = feat
                pred_label = clf.predict(feat_scaled) #(n_token_in_token_span,)
                layer_results[lay_idx][span_label]['y_pred'].append(pred_label)
                layer_results[lay_idx][span_label]['y_true'].append(np.full_like(pred_label, 
                    fill_value=int(sample[args.label_feature])))
                # Projection on probe direction (in scaled space)
                proj_val = np.dot(feat_scaled, w_unit) #(n_token_in_token_span,)
                layer_results[lay_idx][span_label]['proj'].append(proj_val)

        if (samp_idx + 1) % 100 == 0 or samp_idx == n_samples - 1:
            print(f"  Processed {samp_idx + 1}/{n_samples} samples for layers {layer_indices}")


    # ---- Step 8: compute accuracy + average projection magnitude across samples ----

    n_layers_used = len(layer_indices)
    acc_matrix, mag_matrix = compute_avg_acc_mag(
                n_layers_used, span_labels, layer_indices, layer_results)

    # Save matrices
    x_labels = span_labels
    results_dir = os.path.join(base_dir, output_dub_dir)
    save_matrices(results_dir, layer_indices, acc_matrix, mag_matrix, x_labels)

    # ---- Step 9: plot accuracy heatmap [layers × spans] ----
    x_ticks = np.arange(n_spans)
    y_ticks = np.arange(n_layers_used)
    plot_acc(acc_matrix, x_ticks, y_ticks, 
            layer_indices, n_layers_used, n_spans, 
            results_dir, x_labels)

    # ---- Step 10: plot projection magnitude heatmap [layers × spans] ----
    plot_magnitude(mag_matrix, x_ticks, y_ticks, 
                        layer_indices, n_layers_used, n_spans, 
                        results_dir, x_labels)
    print("Done.")

if __name__ == "__main__":
    main()
