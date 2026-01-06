import argparse
import json
import os
import logging

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
from myutilities import save_updated_json
from myutilities import load_probes
from myutilities import save_layer_results 
from myutilities import compute_avg_acc_mag
from myutilities import save_matrices
from myutilities import plot_acc 
from myutilities import plot_magnitude
from myutilities import SPAN_LABELS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(filename)s:%(funcName)s:%(lineno)d: %(message)s]'
)

logger = logging.getLogger(__name__)

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
        "--seed",
        type=int,
        default=45,
        help="Random seed (for reproducibility if needed)",
    )
    parser.add_argument(
        "--probe_base_dir", 
        type=str, 
        default="_datasets/HPC/",
        ) 
    parser.add_argument(
        "--probe_type", 
        type=str, 
        default='normalized',
        help='inf: un-normalized, for activation intervention; normalized: normalized, for information probing'
    ) 

    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    print(args)
    target_file_name= "imdb_sms_interval_1_pairs_with_activations.json"
    input_base_dir = os.path.join(args.data_dir, args.model_name)
    output_base_dir = os.path.join(args.data_dir, args.model_name, 'cross_probing')
    tasks = ['sen_w_t1', 'sen_w_t2', 'sen_w_b']
    data_all = {}
    for task in tasks:
        target_file = str(task) + "_" + target_file_name
        json_path = os.path.join(input_base_dir, task, target_file)
        print(f"Loading metadata from {json_path}")
        data_all[task]=load_metadata(json_path)
    # print(data_all.keys(), type(data_all.keys()))
    n_samples = len(data_all[list(data_all.keys())[0]])
    print(f"Loaded {n_samples} samples.")

    # Determine layers
    first_hidden_path = data_all['sen_w_t1'][0]["hidden_states_file"]
    _, n_layers_total, feat_dim = infer_dims(first_hidden_path)
    print(f"Detected n_layers={n_layers_total}, feat_dim={feat_dim}")
    
    layer_indices = list(range(n_layers_total))
    print(f"Will use probes for layers: {layer_indices}")

    # Load tokenizer
    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    # ---- Step 5: compute token id spans and update JSON ----
    print("Computing token spans per sample...")
    for task in tasks:
        for i, sample in enumerate(data_all[task]):
            token_span_ranges, input_ids = compute_token_spans_for_sample(args.model_name, sample, tokenizer, task)
            sample["token_spans"] = token_span_ranges
            # Optionally also store tokenized length
            sample["prompt_num_tokens"] = len(input_ids)
            if (i + 1) % 100 == 0 or i == n_samples - 1:
                print(f"  Processed {i + 1}/{n_samples} samples for token spans")
        
        target_file = str(task) + "_" + target_file_name
        # Save updated JSON with token spans
        json_with_spans_dir = os.path.join(output_base_dir, task)
        os.makedirs(json_with_spans_dir, exist_ok=True)
        json_with_spans_path = os.path.join(
            json_with_spans_dir,
            target_file.replace(".json", "_with_spans.json"))
        save_updated_json(data_all[task], json_with_spans_path)

    # ---- Step 6 & 7: average pooling per span, apply probes ----
    # Prepare global containers for accuracy + projection magnitude

    output_sub_dir = ''
    probe_sub_dir = ''
    if args.probe_type == 'normalized':
        probe_sub_dir = "linear_probes_logreg_normalized"
        output_sub_dir = "span_probe_results_logreg_normalized"
    else:
        probe_sub_dir = "linear_probes_logreg_inf"
        output_sub_dir = "span_probe_results_logreg_inf"
    # Load all the needed probes
    all_probes = {}
    for task in tasks[:2]:
        all_probes[task] = load_probes(
            args, layer_indices, probe_sub_dir, task)
    
    # process all the samples
    for task in tasks:
        data = data_all[task]
        n_spans = len(SPAN_LABELS)-1 if task=='sen_w_b' else len(SPAN_LABELS)
        x_labels = []
        if task == 'sen_w_t1':
            x_labels = SPAN_LABELS
        elif task == 'sen_w_t2':
            x_labels = SPAN_LABELS
        else:
            x_labels = SPAN_LABELS[:4] + SPAN_LABELS[-1:]
        logger.info(f'Task of datasets: {task}; Span labels {x_labels}')
            
        for probe_task in all_probes:
            print(f'Apply probes of task {probe_task} to activations of task {task}')
            if probe_task == 'sen_w_t1':
                label_feature='label_context_1'
            elif probe_task == 'sen_w_t2':
                label_feature='label_context_2'
            else:
                label_feature=''

            probes = all_probes[probe_task]
            layer_results = {}
            for l_idx in range(n_layers_total):
                layer_results[l_idx] = {
                    "y_true": np.zeros((n_samples,), dtype=np.int32),
                    "y_pred": np.zeros((n_spans, n_samples), dtype=np.int32),
                    "proj": np.zeros((n_spans, n_samples), dtype=np.float32),
                }

            for samp_idx, sample in enumerate(data):
                hidden_path = sample["hidden_states_file"]

                hs = np.load(hidden_path, mmap_mode="r")  # [all_tokens, n_layers, feat_dim]
                all_tokens = hs.shape[0]
                # print('np.load', hidden_path)
                token_spans = sample["token_spans"]
                prompt_num_tokens = sample.get("prompt_num_tokens", all_tokens)
                prompt_num_tokens = min(prompt_num_tokens, all_tokens)
                for li, lay_idx in enumerate(layer_indices):
                    layer_results[lay_idx]['y_true'][samp_idx] = int(sample[label_feature])
                    # print('Probes from layer', lay_idx)
                    scaler = probes[lay_idx]["scaler"]
                    clf = probes[lay_idx]["classifier"]
                    w = clf.coef_.reshape(-1)
                    w_norm = np.linalg.norm(w) + 1e-9
                    w_unit = w / w_norm
                    for s_idx, span_label in enumerate(x_labels):
                        start, end = token_spans[span_label]

                        # Clip into prompt token range, ignore generated tokens
                        start_clipped = max(0, min(start, prompt_num_tokens))
                        end_clipped = max(0, min(end, prompt_num_tokens))

                        # Average pooling across tokens of this span
                        span_reps = hs[start_clipped:end_clipped, lay_idx, :]  # [n_span_tokens, feat_dim]
                        feat = span_reps.mean(axis=0)  # [feat_dim]
                        # for debugging
                        if i < 2:
                            print('hs.shape:', hs.shape)
                            print(f'Check: label: {span_label}; range{token_spans[span_label]}')
                            tokens_in_span = tokenizer([sample['wrapped_prompt']], 
                                return_tensors="pt", padding=False, 
                                truncation=True)['input_ids'][0][start_clipped:end_clipped]
                            print(f"tokens in this span: {tokenizer.convert_ids_to_tokens(tokens_in_span)}")

                        # Apply probe: scale -> predict -> projection
                        try:
                            check_is_fitted(scaler)
                            feat_scaled = scaler.transform(feat.reshape(1, -1))  # [BS==1, feat_dim]
                        except NotFittedError as exc:
                            # print(f"Note scaler is not fitted yet.")
                            feat_scaled = feat.reshape(1, -1)
                        pred_label = clf.predict(feat_scaled)
                        layer_results[lay_idx]['y_pred'][s_idx, samp_idx] = pred_label[0]

                        # Projection on probe direction (in scaled space)
                        proj_val = float(np.dot(feat_scaled[0], w_unit))
                        # proj[s_idx, i] = proj_val
                        layer_results[lay_idx]['proj'][s_idx, samp_idx] = proj_val

                if (samp_idx + 1) % 100 == 0 or samp_idx == n_samples - 1:
                    print(f"  Processed {samp_idx + 1}/{n_samples} samples for layers {layer_indices}")

            # Save per-layer results
            save_layer_results(layer_results, output_base_dir, task, 
                               probe_task, output_sub_dir, layer_indices, x_labels)

            # ---- Step 8: compute accuracy + average projection magnitude across samples ----
            n_layers_used = len(layer_indices)
            acc_matrix, mag_matrix = compute_avg_acc_mag(
                n_layers_used, n_spans, layer_indices, layer_results)

            # Save matrices
            results_dir = os.path.join(output_base_dir, task, probe_task, output_sub_dir)
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


if __name__ == "__main__":
    main()
