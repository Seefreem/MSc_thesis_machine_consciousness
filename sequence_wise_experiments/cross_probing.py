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


SPAN_LABELS = [
    "context1_prefix",  # "context 1: "
    "context1",         # "{c1}; \n"
    "context2_prefix",  # "context 2: "
    "context2",         # "{c2}.\n "
    "task",             # "Task: sentiment analysis ... Sentiment of context 1 is "
    "blank",            # "_"
]

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


def set_seed(seed: int):
    np.random.seed(seed)


def load_metadata(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def infer_dims(first_hidden_path: str):
    hs = np.load(first_hidden_path, mmap_mode="r")
    n_tokens, n_layers, feat_dim = hs.shape
    return n_tokens, n_layers, feat_dim


def compute_token_spans_for_sample(sample, tokenizer, task):
    wrapped_prompt = sample["wrapped_prompt"]
    s0 = "context 1:"
    s1 = "; \n"
    s2 = "context 2:"
    s3 = ".\n "
    s4 = ''
    s5 = "_"
    if task == 'sen_w_t1':
        s4 = (
            " Task: sentiment analysis of context 1, positive or negative.\nSentiment of context 1 is"
        )    
    elif task == 'sen_w_t2':
        s4 = (
            " Task: classify SMS incontext 2 as spam or ham (not a spam).\nContext 2 is classified as"
        )
    elif task == 'sen_w_b':
        pass
    else:
        raise ValueError(f"Unhandled task: {task}")
        
    token_span_ranges = {}
    token_ids_wp = tokenizer(wrapped_prompt, padding=False)['input_ids']
    token_ids_s0 = tokenizer(s0, add_special_tokens=False)['input_ids']
    token_ids_s2 = tokenizer(s2, add_special_tokens=False)['input_ids']
    token_ids_s4 = tokenizer(s4, add_special_tokens=False)['input_ids']
    # print(token_ids_wp.shap, token_ids_wp)
    # print(token_ids_s0.shape, token_ids_s0)
    
    if task == 'sen_w_b':
        pieces = [token_ids_s0, token_ids_s2]
        labels = [SPAN_LABELS[0], SPAN_LABELS[2]]
    else:
        pieces = [token_ids_s0, token_ids_s2, token_ids_s4]
        labels = [SPAN_LABELS[0], SPAN_LABELS[2], SPAN_LABELS[4]]
    
    for (span, label) in zip(pieces, labels):
        len_span = len(span)
        for i in range(len(token_ids_wp) - len_span):
            # print(span == token_ids_wp[i:i + len_span])
            # print((span == token_ids_wp[i:i + len_span]))
            if (span == token_ids_wp[i:i + len_span]):
                # print(f'Found the span of {label}')
                start_s0 = i
                end_s0 = i + len_span
                token_span_ranges[label] = [start_s0, end_s0]
                break
    # check 
    token_span_ranges[SPAN_LABELS[1]] = [token_span_ranges[SPAN_LABELS[0]][1], token_span_ranges[SPAN_LABELS[2]][0]]
    token_span_ranges[SPAN_LABELS[5]] = [len(token_ids_wp) - 1, len(token_ids_wp)]
    if task == 'sen_w_b':
        if not (SPAN_LABELS[0] in token_span_ranges.keys() and 
            SPAN_LABELS[2] in token_span_ranges.keys()):
            raise ValueError(f'Key words not found of the sample: {sample}')
        token_span_ranges[SPAN_LABELS[3]] = [token_span_ranges[SPAN_LABELS[2]][1], len(token_ids_wp) - 1]
    else:
        if not (SPAN_LABELS[0] in token_span_ranges.keys() and 
            SPAN_LABELS[2] in token_span_ranges.keys() and 
            SPAN_LABELS[4] in token_span_ranges.keys()):
            raise ValueError(f'Key words not found of the sample: {sample}')
        token_span_ranges[SPAN_LABELS[3]] = [token_span_ranges[SPAN_LABELS[2]][1], token_span_ranges[SPAN_LABELS[4]][0]]
    return token_span_ranges, token_ids_wp

def save_updated_json(data, out_path):
    print(f"Saving updated JSON with token spans to {out_path}")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_probes(args, layer_indices, probe_sub_dir, task):
    all_probes = {}
    for l_idx in layer_indices:
        # Load probe for fold 0
        probe_dir = os.path.join(
            args.probe_base_dir, args.model_name, task, probe_sub_dir, f"layer_{l_idx}"
        )
        probe_path = os.path.join(
            probe_dir, f"logreg_layer{l_idx}_fold0.joblib"
        )
        print(f"  Loading probe from {probe_path}")
        # bundle = joblib.load(probe_path)
        all_probes[l_idx] = joblib.load(probe_path)
    return all_probes


def plot_magnitude(mag_matrix, x_ticks, y_ticks, 
                   layer_indices, n_layers_used, n_spans, 
                   results_dir, x_labels):
    print("Plotting projection magnitude heatmap...")
    fig_mag = plt.figure()
    # Normalize magnitudes between min and max just for visualization
    vmin = float(np.nanmin(mag_matrix))
    vmax = float(np.nanmax(mag_matrix))
    im = plt.imshow(mag_matrix, vmin=vmin, vmax=vmax, aspect="auto")
    plt.colorbar(im, label="Avg |projection|")

    plt.xticks(x_ticks, x_labels, rotation=90, ha="right", fontsize = 7)
    plt.yticks(y_ticks, layer_indices, fontsize = 7)

    # Add magnitude values
    for i in range(n_layers_used):
        for j in range(n_spans):
            val = mag_matrix[i, j]
            # text_color = "white" if (val - vmin) > 0.5 * (vmax - vmin) else "black"
            text_color = "black"
            plt.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color=text_color,
            )

    plt.xlabel("Token span")
    plt.ylabel("Layer index")
    plt.title("Average |projection| per span per layer")
    plt.tight_layout()

    mag_fig_path = os.path.join(results_dir, "span_projection_magnitude_heatmap.png")
    plt.savefig(mag_fig_path, dpi=200)
    plt.close(fig_mag)
    print(f"Projection magnitude heatmap saved to: {mag_fig_path}")

def plot_acc(acc_matrix, x_ticks, y_ticks, 
             layer_indices, n_layers_used, n_spans, 
             results_dir, x_labels):
    print("Plotting accuracy heatmap...")
    fig_acc = plt.figure()
    im = plt.imshow(acc_matrix, vmin=0.0, vmax=1.0, aspect="auto")
    plt.colorbar(im, label="Accuracy")

    plt.xticks(x_ticks, x_labels, rotation=90, ha="right", fontsize = 7)
    plt.yticks(y_ticks, layer_indices, fontsize = 7)

    # Add accuracy values in each cell
    for i in range(n_layers_used):
        for j in range(n_spans):
            val = acc_matrix[i, j]
            # text_color = "white" if val > 0.5 else "black"
            text_color = "black"
            plt.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color=text_color,
            )

    plt.xlabel("Token span")
    plt.ylabel("Layer index")
    plt.title("Average accuracy per span per layer")
    plt.tight_layout()

    acc_fig_path = os.path.join(results_dir, "span_accuracy_heatmap.png")
    plt.savefig(acc_fig_path, dpi=200)
    plt.close(fig_acc)
    print(f"Accuracy heatmap saved to: {acc_fig_path}")

def save_matrices(results_dir, layer_indices, acc_matrix, mag_matrix, span_labels):
    output_file_name = os.path.join(results_dir, "span_accuracy_and_magnitude_all_layers.npz")
    print('Save the accuracy and projected magnitude of all layers to file', output_file_name)
    os.makedirs(results_dir, exist_ok=True)
    np.savez(
        output_file_name,
        layer_indices=np.array(layer_indices),
        span_labels=np.array(span_labels),
        acc_matrix=acc_matrix,
        mag_matrix=mag_matrix,
    )

def compute_avg_acc_mag(n_layers_used, n_spans, layer_indices, layer_results):
    print("Computing accuracy and average projection magnitude matrices...")
    acc_matrix = np.zeros((n_layers_used, n_spans), dtype=np.float32)
    mag_matrix = np.zeros((n_layers_used, n_spans), dtype=np.float32)

    for li, layer_idx in enumerate(layer_indices):
        res = layer_results[layer_idx]
        y_true = res["y_true"]            # [n_samples]
        y_pred = res["y_pred"]            # [n_spans, n_samples]
        proj = res["proj"]                # [n_spans, n_samples]

        for s_idx in range(n_spans):
            preds = y_pred[s_idx]
            # Accuracy for this span & layer
            acc = accuracy_score(y_true, preds)
            acc_matrix[li, s_idx] = acc

            # Average magnitude of projection (abs) across samples
            proj_vals = proj[s_idx]
            # Ignore NaNs if any
            mag = np.nanmean(np.abs(proj_vals))
            mag_matrix[li, s_idx] = mag
    return acc_matrix, mag_matrix

def save_layer_results(layer_results, base_dir, task, probe_task, output_sub_dir, layer_indices, span_labels):
    for li, lay_idx in enumerate(layer_indices):
        layer_out_dir = os.path.join(
            base_dir, task, probe_task, output_sub_dir, f"layer_{lay_idx}"
        )
        os.makedirs(layer_out_dir, exist_ok=True)
        npz_path = os.path.join(layer_out_dir, f"span_results_layer{lay_idx}.npz")
        print(f"  Saving per-layer span results to {npz_path}")
        np.savez(
            npz_path,
            span_labels=np.array(span_labels),
            y_true=layer_results[lay_idx]["y_true"],
            y_pred=layer_results[lay_idx]["y_pred"],
            proj=layer_results[lay_idx]["proj"],
        )

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
            token_span_ranges, input_ids = compute_token_spans_for_sample(sample, tokenizer, task)
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
