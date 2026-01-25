import json
import os
import torch
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

SPAN_LABELS = [
    "context1_prefix",  # "context 1: "
    "context1",         # "{c1}; \n"
    "context2_prefix",  # "context 2: "
    "context2",         # "{c2}.\n "
    "task",             # "Task: sentiment analysis ... Sentiment of context 1 is "
    "last_token",            # "_"
]

SPAN_LABELS_EXP2 = [
    "context_prefix",  # "context: "
    "context",         # "{c1}. \n"
    "task",             # "Task: Binary humor classification ... Answer: The given context is"
    "last_token",            # "_"
]

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

def compute_avg_acc_mag(n_layers_used, span_labels, layer_indices, layer_results):
    print("Computing accuracy and average projection magnitude matrices...")
    acc_matrix = np.zeros((n_layers_used, len(span_labels)), dtype=np.float32)
    mag_matrix = np.zeros((n_layers_used, len(span_labels)), dtype=np.float32)
    # layer_results[lay_idx][span_label]['y_true']
    for li, layer_idx in enumerate(layer_indices):
        l_res = layer_results[layer_idx]
        for s_idx, s_label in enumerate(span_labels):
            y_true = np.concatenate(l_res[s_label]["y_true"])            
            y_pred = np.concatenate(l_res[s_label]["y_pred"])            
            proj = np.concatenate(l_res[s_label]["proj"])                

            # Accuracy for this span & layer
            acc = accuracy_score(y_true, y_pred)
            acc_matrix[li, s_idx] = acc

            # Average magnitude of projection (abs) across samples
            # Ignore NaNs if any
            mag = np.nanmean(np.abs(proj))
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

def save_updated_json(data, out_path):
    print(f"Saving updated JSON with token spans to {out_path}")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def compute_token_spans_for_sample(model, sample, tokenizer, task):
    wrapped_prompt = sample["wrapped_prompt"]
    s0 = "context 1:"
    s1 = "; \n"
    s2 = "context 2:"
    s3 = ".\n "
    s4 = ''
    s5 = "_"
    if task == 'sen_w_t1':
        if 'gemma' in model:
            s4 = (
                " Task: sentiment analysis of context 1; the only labels: positive or negative.\nSentiment of context 1 is" # ['▁**']
            )    
        elif 'llama' in model:
            s4 = (
                " Task: sentiment analysis of context 1, positive or negative.\nSentiment of context 1 is"
            )    
    elif task == 'sen_w_t2':
        if 'gemma' in model:
            s4 = (
                " Task: classify SMS in context 2 as spam or ham (not a spam).\nContext 2 is classified as"
            )    
        elif 'llama' in model:
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
def compute_token_spans_for_exp2_samples(model, sample, tokenizer, task):
    wrapped_prompt = sample["wrapped_prompt"]
    s0 = "Context:"
    s2=''
    if task == 'lay_w_t1':
        if 'gemma' in model:
            s2 = "Task: Binary humor classification of the given context (humorous or not humorous).\nAnswer: The given context is" # ['▁**'] 
        elif 'llama' in model:
            s2 = "Task: Binary humor classification of the given context (humorous or not humorous).\nAnswer: The given context is" 
    elif task == 'lay_w_t2':
        if 'gemma' in model:
            s2 = (
                "Task: Binary offense classification of the given context (offensive or not offensive).\nAnswer: The given context is"
            )    
        elif 'llama' in model:
            s2 = (
                "Task: Binary offense classification of the given context (offensive or not offensive).\nAnswer: The given context is"
            )
    elif task == 'lay_w_b':
        pass
    else:
        raise ValueError(f"Unhandled task: {task}")
        
    token_span_ranges = {}
    token_ids_wp = tokenizer(wrapped_prompt, padding=False)['input_ids']
    token_ids_s0 = tokenizer(s0, add_special_tokens=False)['input_ids']
    token_ids_s2 = tokenizer(s2, add_special_tokens=False)['input_ids']
    # print('tokens_wp\n', tokenizer.convert_ids_to_tokens(token_ids_wp))
    # print('token_ids_wp\n', len(token_ids_wp), token_ids_wp)
    # print('token_ids_s0\n', len(token_ids_s0), token_ids_s0)
    # print('tokens_s0\n', tokenizer.convert_ids_to_tokens(token_ids_s0))
    # print('token_ids_s2\n', len(token_ids_s2), token_ids_s2)
    # print('tokens_s2\n', tokenizer.convert_ids_to_tokens(token_ids_s2))
    
    if task == 'lay_w_b':
        pieces = [token_ids_s0]
        labels = [SPAN_LABELS_EXP2[0]]
    else:
        pieces = [token_ids_s0, token_ids_s2]
        labels = [SPAN_LABELS_EXP2[0], SPAN_LABELS_EXP2[2]]
    
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
    # token_span_ranges[SPAN_LABELS_EXP2[1]] = [token_span_ranges[SPAN_LABELS_EXP2[0]][1], token_span_ranges[SPAN_LABELS_EXP2[2]][0]]
    token_span_ranges[SPAN_LABELS_EXP2[3]] = [len(token_ids_wp) - 1, len(token_ids_wp)]
    if task == 'lay_w_b':
        if not (SPAN_LABELS_EXP2[0] in token_span_ranges.keys() and 
            SPAN_LABELS_EXP2[3] in token_span_ranges.keys()):
            raise ValueError(f'Key words not found of the sample: {sample}')
        token_span_ranges[SPAN_LABELS_EXP2[1]] = [token_span_ranges[SPAN_LABELS_EXP2[0]][1], token_span_ranges[SPAN_LABELS_EXP2[3]][0]]
    else:
        if not (SPAN_LABELS_EXP2[0] in token_span_ranges.keys() and 
            SPAN_LABELS_EXP2[2] in token_span_ranges.keys() and 
            SPAN_LABELS_EXP2[3] in token_span_ranges.keys()):
            raise ValueError(f'Key words not found of the sample: {sample}')
        token_span_ranges[SPAN_LABELS_EXP2[1]] = [token_span_ranges[SPAN_LABELS_EXP2[0]][1], token_span_ranges[SPAN_LABELS_EXP2[2]][0]]
    # print('token_span_ranges\n', token_span_ranges)
    return token_span_ranges, token_ids_wp

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


def wrap_example(model, context_1: str, context_2: str, template_type: str) -> str:
    """Create the prompt string for one example."""
    PROMPT_TEMPLATE = ""
    if template_type == 'sen_w_t1':
        if 'gemma' in model:
            PROMPT_TEMPLATE = (
                "context 1: {c1}; \ncontext 2: {c2}.\n "
                "Task: sentiment analysis of context 1; the only labels: positive or negative.\n"
                "Sentiment of context 1 is **"
                # "Among the labels of positive and negative, the sentiment label of context 1 is: "
            )
        elif 'llama' in model:
            PROMPT_TEMPLATE = (
                "context 1: {c1}; \ncontext 2: {c2}.\n "
                "Task: sentiment analysis of context 1, positive or negative.\n"
                "Sentiment of context 1 is _"
                # "Among the labels of positive and negative, the sentiment label of context 1 is: "
            )
        PROMPT_TEMPLATE = PROMPT_TEMPLATE.format(c1=context_1, c2=context_2)
    elif template_type == 'sen_w_t2':
        if 'gemma' in model:
            PROMPT_TEMPLATE = (
                "context 1: {c1}; \ncontext 2: {c2}.\n "
                "Task: classify SMS in context 2 as spam or ham (not a spam).\n"
                "Context 2 is classified as **"
            )
        elif 'llama' in model:
            PROMPT_TEMPLATE = (
                "context 1: {c1}; \ncontext 2: {c2}.\n "
                "Task: classify SMS incontext 2 as spam or ham (not a spam).\n"
                "Context 2 is classified as _"
            )
        PROMPT_TEMPLATE = PROMPT_TEMPLATE.format(c1=context_1, c2=context_2)
    elif template_type == 'sen_w_b':
        if 'gemma' in model:
            PROMPT_TEMPLATE = (
                "context 1: {c1}; \ncontext 2: {c2}.\n **"
            )
        elif 'llama' in model:
            PROMPT_TEMPLATE = (
                "context 1: {c1}; \ncontext 2: {c2}.\n _"
            )
        PROMPT_TEMPLATE = PROMPT_TEMPLATE.format(c1=context_1, c2=context_2)
    elif template_type == 'lay_w_t1':
        if 'gemma' in model:
            PROMPT_TEMPLATE = (
                "Context: {c1}. \n"
                "Task: Binary humor classification of the given context (humorous or not humorous).\n"
                "Answer: The given context is **"
            )
        elif 'llama' in model:
            PROMPT_TEMPLATE = (
                "Context: {c1}. \n"
                "Task: Binary humor classification of the given context (humorous or not humorous).\n"
                "Answer: The given context is _"
            )
        PROMPT_TEMPLATE = PROMPT_TEMPLATE.format(c1=context_1)
    elif template_type == 'lay_w_t2':
        if 'gemma' in model:
            PROMPT_TEMPLATE = (
                "Context: {c1}. \n"
                "Task: Binary offense classification of the given context (offensive or not offensive).\n"
                "Answer: The given context is **"
            )
        elif 'llama' in model:
            PROMPT_TEMPLATE = (
                "Context: {c1}. \n"
                "Task: Binary offense classification of the given context (offensive or not offensive).\n"
                "Answer: The given context is _"
            )
        PROMPT_TEMPLATE = PROMPT_TEMPLATE.format(c1=context_1)
    elif template_type == 'lay_w_b':
        if 'gemma' in model:
            PROMPT_TEMPLATE = (
                "Context: {c1}. **"
            )
        elif 'llama' in model:
            PROMPT_TEMPLATE = (
                "Context: {c1}. _"
            )
        PROMPT_TEMPLATE = PROMPT_TEMPLATE.format(c1=context_1)
    elif template_type == 'selective_attention':
        raise ValueError(f"Unhandled template type {template_type}")
    else:
        raise ValueError(f"Unknown template type {template_type}")
    return PROMPT_TEMPLATE



def save_activations(path: str, arr: np.ndarray):
    # arr16 = arr.astype(np.float16)     # or np.float32 -> np.float16
    # np.savez_compressed(path, arr=arr16)
    np.save(path, arr=arr)

def load_activations(path: str) -> np.ndarray:
    # data = np.load(path)
    # arr16 = data["arr"]
    # return arr16.astype(np.float32)    # back to float32 if you want
    return np.load(path)

def setup_model_and_tokenizer(model_name: str):
    """Load tokenizer and model from HuggingFace."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        # Llama-style models often have no pad token set
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.config.temperature = 1.0
    model.config.top_p=1.0
    model.eval()
    return tokenizer, model

def register_hooks(model):
    """
    Register hooks on MLP and attention output projection (o_proj) modules.

    Returns:
        mlp_acts, attn_block_outs: lists of tensors per layer; they will be
        filled on each forward pass.
    """
    # Llama-style: model.model.layers is a list of decoder layers
    decoder_layers = model.model.layers
    num_layers = len(decoder_layers)
    print(f"num_layers: {num_layers}")

    # mlp_acts: List[torch.Tensor] = [None] * num_layers
    # attn_block_outs: List[torch.Tensor] = [None] * num_layers

    mlp_acts: List[List] = [[]] * num_layers
    attn_block_outs: List[List] = [[]] * num_layers

    def make_mlp_hook(layer_idx: int):
        def hook(module, input, output):
            # module: the module per se; input: the input of this module; output: the output of this module
            # output: [batch, seq_len, hidden_dim]
            mlp_acts[layer_idx].append(output.detach().to("cpu"))
        return hook

    def make_attn_hook(layer_idx: int):
        # hook on the output projection (combined multi-head attention output)
        def hook(module, input, output):
            # output: [batch, seq_len, hidden_dim]
            attn_block_outs[layer_idx].append(output.detach().to("cpu"))
        return hook

    for i, layer in enumerate(decoder_layers):
        layer.mlp.register_forward_hook(make_mlp_hook(i))
        layer.self_attn.o_proj.register_forward_hook(make_attn_hook(i))

    return mlp_acts, attn_block_outs

def tuple_to_numpy_all_tokens(
    data: Tuple[Tuple[torch.Tensor, ...], ...]
) -> np.ndarray:
    """
    Transform a tuple(new_tokens, n_layers, torch.Tensor) into a numpy array
    of shape [all_tokens, n_layers, hidden_size].

    Args:
        data: Nested tuple with shape:
              data[token_idx][layer_idx] -> torch.Tensor of shape
              (batch_size, seq_len, hidden_size),
              where:
                - For token_idx == 0: seq_len == n_input_tokens
                - For token_idx > 0:  seq_len == 1

    Returns:
        np.ndarray of shape [all_tokens, n_layers, hidden_size],
        where:
            all_tokens = n_input_tokens + (new_tokens - 1)
    """
    # Basic structure
    new_tokens = len(data)
    if new_tokens == 0:
        raise ValueError("Empty data tuple.")

    n_layers = len(data[0])
    if n_layers == 0:
        raise ValueError("No layers found in data[0].")

    # Inspect first tensor for dimensions
    first_tensor = data[0][0]
    if first_tensor.ndim != 3:
        raise ValueError(f"Expected tensor with 3 dims (BS, seq_len, hidden), got {first_tensor.shape}")
    bs, n_input_tokens, hidden_size = first_tensor.shape
    if bs != 1:
        raise ValueError(f"Function assumes batch_size=1, got batch_size={bs}")

    # Total number of tokens = all input tokens + one per subsequent new token
    all_tokens = n_input_tokens + (new_tokens - 1)

    # Preallocate numpy array
    out = np.empty((all_tokens, n_layers, hidden_size), dtype=np.float32)

    # 1) Fill in representations for all input tokens from the first step
    for layer_idx in range(n_layers):
        t = data[0][layer_idx]  # shape: [1, n_input_tokens, hidden_size]
        if t.shape[1] != n_input_tokens:
            raise ValueError("Inconsistent seq_len for first new token across layers.")
        out[:n_input_tokens, layer_idx, :] = t[0].detach().cpu().float().numpy()# .astype(np.float16)

    # 2) Append representations of each subsequent new token
    current_token_idx = n_input_tokens
    for token_idx in range(1, new_tokens):
        for layer_idx in range(n_layers):
            t = data[token_idx][layer_idx]  # shape: [1, 1, hidden_size] (by your description)
            if t.shape[1] != 1:
                raise ValueError(
                    f"Expected seq_len=1 for token_idx={token_idx}, "
                    f"layer_idx={layer_idx}, got {t.shape}"
                )
            # Take the last (or only) position
            out[current_token_idx, layer_idx, :] = t[0, -1].detach().cpu().float().numpy()# .astype(np.float16)
        current_token_idx += 1
    # print('check1====')
    # print(data[0][0][0, 0,:])
    # print(out[0, 0, :])
    # print('check2====')
    # print(data[-1][-1][0, 0,:])
    # print(out[-1, -1, :])
    return out

def attn_tuple_to_padded_numpy(
    data: Tuple[Tuple[torch.Tensor, ...], ...]
) -> np.ndarray:
    """
    Transform a tuple(new_tokens, layers, torch.Tensor) of attention scores into a
    zero-left-padded numpy array of shape:
        (new_tokens, n_layers, n_heads, max_seq_len)

    Input structure:
        data[token_idx][layer_idx] -> torch.Tensor of shape
            (batch_size, n_heads, 1, seq_len)

    Zero-left-padding means:
        For each (token, layer, head), its attention vector of length seq_len
        is right-aligned in the last dimension, with zeros on the left.

    Returns:
        np.ndarray of shape (new_tokens, n_layers, n_heads, max_seq_len)
    """
    new_tokens = len(data)
    if new_tokens == 0:
        raise ValueError("Empty data tuple.")

    n_layers = len(data[0])
    if n_layers == 0:
        raise ValueError("No layers found in data[0].")

    # Check basic tensor shape and collect max seq_len
    max_seq_len = 0
    n_heads = None

    for t_idx in range(new_tokens):
        if len(data[t_idx]) != n_layers:
            raise ValueError("Inconsistent number of layers across tokens.")
        for l_idx in range(n_layers):
            tensor = data[t_idx][l_idx]
            if tensor.ndim != 4:
                raise ValueError(
                    f"Expected tensor with 4 dims (BS, heads, 1, seq_len), got {tensor.shape}"
                )
            bs, h, one_dim, seq_len = tensor.shape
            if bs != 1:
                raise ValueError(f"Function assumes batch_size=1, got {bs}")
            # if one_dim != 1:
            #     raise ValueError(
            #         f"Expected third dim to be 1 (got {one_dim}) in tensor {t_idx}, layer {l_idx}"
            #     )
            if n_heads is None:
                n_heads = h
            elif n_heads != h:
                raise ValueError("Inconsistent number of heads across tensors.")
            if seq_len > max_seq_len:
                max_seq_len = seq_len

    if n_heads is None:
        raise ValueError("Could not infer number of heads.")

    # Preallocate output: zero-initialized for left-padding
    out = np.zeros(
        (new_tokens, n_layers, n_heads, max_seq_len),
        dtype=np.float32,
    )

    # Fill with right-aligned attention scores
    for t_idx in range(new_tokens):
        for l_idx in range(n_layers):
            tensor = data[t_idx][l_idx]        # [1, n_heads, 1, seq_len]
            _, _, _, seq_len = tensor.shape

            # Extract attention: [n_heads, seq_len]
            attn = tensor[0, :, -1, :].detach().cpu().float().numpy()# .astype(np.float16)

            # Right-align: put it in the last seq_len positions
            out[t_idx, l_idx, :, max_seq_len - seq_len:] = attn
    # print('check1====')
    # print(data[0][0][0, 0, -1,:])
    # print(out[0, 0, 0, :])
    # print('check2====')
    # print(data[-1][-1][0, -1, 0,:])
    # print(out[-1, -1, -1, :])
    return out

def layer_token_tuple_to_numpy(
    data: Tuple[Tuple[torch.Tensor, ...], ...]
) -> np.ndarray:
    """
    Transform data with structure:
        data[layer_idx][token_idx] -> tensor of shape (BS=1, tokens_t, activation_dim)

    into a numpy array of shape:
        (total_tokens, n_layers, activation_dim)

    Assumptions:
      - BS == 1
      - First tuple dimension = layers
      - Second tuple dimension = new generated tokens / time steps
      - For each token_idx, tokens_t can be 1 or >1, but is consistent across layers.
    """
    n_layers = len(data)
    if n_layers == 0:
        raise ValueError("Empty data (no layers).")

    n_steps = len(data[0])  # number of token-steps
    if n_steps == 0:
        raise ValueError("No token steps in data[0].")

    # Check all layers have same number of steps
    for layer_idx in range(n_layers):
        if len(data[layer_idx]) != n_steps:
            raise ValueError("Inconsistent number of token steps across layers.")

    # Infer activation_dim and tokens per step from layer 0
    activation_dim = None
    tokens_per_step = []

    for step_idx in range(n_steps):
        tensor = data[0][step_idx]
        if tensor.ndim != 3:
            raise ValueError(
                f"Expected tensor with 3 dims (BS, tokens, activation_dim), got {tensor.shape}"
            )
        bs, tokens_t, act_dim_t = tensor.shape
        if bs != 1:
            raise ValueError(f"Expected batch_size=1, got {bs}")

        if activation_dim is None:
            activation_dim = act_dim_t
        elif activation_dim != act_dim_t:
            raise ValueError("Inconsistent activation_dim across steps in layer 0.")

        tokens_per_step.append(tokens_t)

    if activation_dim is None:
        raise ValueError("Could not infer activation_dim.")

    total_tokens = sum(tokens_per_step)

    # Allocate output: (total_tokens, n_layers, activation_dim)
    out = np.empty((total_tokens, n_layers, activation_dim), dtype=np.float32)

    # Fill: iterate over layers and steps, share the same token offset across layers
    offset = 0
    for step_idx, tokens_t in enumerate(tokens_per_step):
        for layer_idx in range(n_layers):
            tensor = data[layer_idx][step_idx]  # (1, tokens_t, activation_dim)
            if tensor.shape != (1, tokens_t, activation_dim):
                raise ValueError(
                    f"Inconsistent shape at layer {layer_idx}, step {step_idx}: "
                    f"got {tensor.shape}, expected (1, {tokens_t}, {activation_dim})"
                )
            arr = tensor[0].detach().cpu().float().numpy()# .astype(np.float16)  # (tokens_t, activation_dim)
            out[offset : offset + tokens_t, layer_idx, :] = arr

        offset += tokens_t
    # print('check1====')
    # print(data[0][0][0, 0, :])
    # print(out[0,0,:])
    # print('check2====')
    # print(data[-1][-1][0, 0, :])
    # print(out[-1,-1,:])
    return out