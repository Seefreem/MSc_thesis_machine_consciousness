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


PROMPT_TEMPLATE = (
    "context 1: {c1}; \ncontext 2: {c2}.\n "
    "Task: sentiment analysis of context 1, positive or negative.\n"
    "Sentiment of context 1 is _"
)

SPAN_LABELS = [
    "context1_prefix",  # "context 1: "
    "context1",         # "{c1}; \n"
    "context2_prefix",  # "context 2: "
    "context2",         # "{c2}.\n "
    "task",             # "Task: sentiment analysis ... Sentiment of context 1 is "
    "blank",            # "_"
]


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
        default="imdb_sms_interval_1_pairs_with_activations.json",
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
        help='Candidate tasks: sen_w_t1, sen_w_t2, sen_w_b, lay_w_t1, lay_w_t2, lay_w_b, and selective_attention'
        ) 
    parser.add_argument(
        "--probe_base_dir", 
        type=str, 
        default="_dayasets",
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


def build_span_char_ranges(sample): # What is this function for?
    """
    Build character-level span ranges for the wrapped_prompt, as:
    0: "context 1: "
    1: "{c1}; \n"
    2: "context 2: "
    3: "{c2}.\n "
    4: "Task: sentiment analysis of context 1, positive or negative.\nSentiment of context 1 is "
    5: "_"
    Returns:
        span_ranges: list of (start_char, end_char) for each span
        prompt_text: the reconstructed prompt (should equal sample["wrapped_prompt"])
    """
    c1 = sample["context_1"]
    c2 = sample["context_2"]

    s0 = "context 1: "
    s1 = c1 + "; \n"
    s2 = "context 2: "
    s3 = c2 + ".\n "
    s4 = (
        "Task: sentiment analysis of context 1, positive or negative.\n"
        "Sentiment of context 1 is "
    )
    s5 = "_"

    pieces = [s0, s1, s2, s3, s4, s5]
    prompt_text = "".join(pieces)

    # Build cumulative character ranges
    span_ranges = []
    cur = 0
    for seg in pieces:
        start = cur
        end = cur + len(seg)
        span_ranges.append((start, end))
        cur = end
    print('span_ranges', span_ranges)
    return span_ranges, prompt_text

def compute_token_spans_for_sample(sample, tokenizer, task):
    wrapped_prompt = sample["wrapped_prompt"]
    if task == 'sen_w_t1':
        s0 = "context 1:"
        s1 = "; \n"
        s2 = "context 2:"
        s3 = ".\n "
        s4 = (
            " Task: sentiment analysis of context 1, positive or negative.\nSentiment of context 1 is"
        )
        s5 = "_"
        
        token_span_ranges = {}
        token_ids_wp = tokenizer(wrapped_prompt, padding=False)['input_ids']
        token_ids_s0 = tokenizer(s0, add_special_tokens=False)['input_ids']
        token_ids_s2 = tokenizer(s2, add_special_tokens=False)['input_ids']
        token_ids_s4 = tokenizer(s4, add_special_tokens=False)['input_ids']
        # print(token_ids_wp.shap, token_ids_wp)
        # print(token_ids_s0.shape, token_ids_s0)
        
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
        if not (SPAN_LABELS[0] in token_span_ranges.keys() and 
            SPAN_LABELS[2] in token_span_ranges.keys() and 
            SPAN_LABELS[4] in token_span_ranges.keys()):
            ValueError(f'Key words not found of the sample {wrap_example}')

        # print(token_span_ranges)    
        token_span_ranges[SPAN_LABELS[1]] = [token_span_ranges[SPAN_LABELS[0]][1], token_span_ranges[SPAN_LABELS[2]][0]]
        token_span_ranges[SPAN_LABELS[3]] = [token_span_ranges[SPAN_LABELS[2]][1], token_span_ranges[SPAN_LABELS[4]][0]]
        token_span_ranges[SPAN_LABELS[5]] = [len(token_ids_wp) - 1, len(token_ids_wp)]
        # print(token_span_ranges)
        return token_span_ranges, token_ids_wp

        
    else:
        ValueError(f"Unhandled task: {task}")

def save_updated_json(data, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


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
    for i, sample in enumerate(data):
        token_span_ranges, input_ids = compute_token_spans_for_sample(sample, tokenizer, args.task)
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
    print(f"Saving updated JSON with token spans to {json_with_spans_path}")
    save_updated_json(data, json_with_spans_path)

    # ---- Step 6 & 7: average pooling per span, apply probes ----
    # Prepare global containers for accuracy + projection magnitude
    n_spans = len(SPAN_LABELS)
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
    for l_idx in layer_indices:
        # Load probe for fold 0
        probe_dir = os.path.join(
            args.probe_base_dir, args.model_name, args.task, parent_dir, f"layer_{l_idx}"
        )
        probe_path = os.path.join(
            probe_dir, f"logreg_layer{l_idx}_fold0.joblib"
        )
        print(f"  Loading probe from {probe_path}")
        # bundle = joblib.load(probe_path)
        all_probes[l_idx] = joblib.load(probe_path)
        
        layer_results[l_idx] = {
            "y_true": np.zeros((n_samples,), dtype=np.int32),
            "y_pred": np.zeros((n_spans, n_samples), dtype=np.int32),
            "proj": np.zeros((n_spans, n_samples), dtype=np.float32),
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
            layer_results[lay_idx]['y_true'][samp_idx] = int(sample[args.label_feature])
            # print('Probes from layer', lay_idx)
            scaler = all_probes[lay_idx]["scaler"]
            clf = all_probes[lay_idx]["classifier"]
            w = clf.coef_.reshape(-1)
            w_norm = np.linalg.norm(w) + 1e-9
            w_unit = w / w_norm
            for s_idx, span_label in enumerate(SPAN_LABELS):
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

    for li, lay_idx in enumerate(layer_indices):
        # Save per-layer results
        layer_out_dir = os.path.join(
            base_dir, output_dub_dir, f"layer_{lay_idx}"
        )
        os.makedirs(layer_out_dir, exist_ok=True)
        npz_path = os.path.join(layer_out_dir, f"span_results_layer{lay_idx}.npz")
        print(f"  Saving per-layer span results to {npz_path}")
        np.savez(
            npz_path,
            span_labels=np.array(SPAN_LABELS),
            y_true=layer_results[lay_idx]["y_true"],
            y_pred=layer_results[lay_idx]["y_pred"],
            proj=layer_results[lay_idx]["proj"],
        )

    # ---- Step 8: compute accuracy + average projection magnitude across samples ----
    print("Computing accuracy and average projection magnitude matrices...")

    n_layers_used = len(layer_indices)
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

    # Save matrices
    results_dir = os.path.join(base_dir, output_dub_dir)
    os.makedirs(results_dir, exist_ok=True)
    np.savez(
        os.path.join(results_dir, "span_accuracy_and_magnitude_all_layers.npz"),
        layer_indices=np.array(layer_indices),
        span_labels=np.array(SPAN_LABELS),
        acc_matrix=acc_matrix,
        mag_matrix=mag_matrix,
    )

    # ---- Step 9: plot accuracy heatmap [layers × spans] ----
    print("Plotting accuracy heatmap...")
    fig_acc = plt.figure()
    im = plt.imshow(acc_matrix, vmin=0.0, vmax=1.0, aspect="auto")
    plt.colorbar(im, label="Accuracy")

    x_ticks = np.arange(n_spans)
    y_ticks = np.arange(n_layers_used)
    plt.xticks(x_ticks, SPAN_LABELS, rotation=90, ha="right", fontsize = 7)
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

    # ---- Step 10: plot projection magnitude heatmap [layers × spans] ----
    print("Plotting projection magnitude heatmap...")
    fig_mag = plt.figure()
    # Normalize magnitudes between min and max just for visualization
    vmin = float(np.nanmin(mag_matrix))
    vmax = float(np.nanmax(mag_matrix))
    im = plt.imshow(mag_matrix, vmin=vmin, vmax=vmax, aspect="auto")
    plt.colorbar(im, label="Avg |projection|")

    plt.xticks(x_ticks, SPAN_LABELS, rotation=90, ha="right", fontsize = 7)
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

    print("Done.")
    print(f"Accuracy heatmap saved to: {acc_fig_path}")
    print(f"Projection magnitude heatmap saved to: {mag_fig_path}")


if __name__ == "__main__":
    main()
