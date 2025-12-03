import json
import os
import torch
import numpy as np

from pathlib import Path
from typing import Dict, Any, List, Tuple
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

def save_activations(path: str, arr: np.ndarray):
    arr16 = arr.astype(np.float16)     # or np.float32 -> np.float16
    # np.savez_compressed(path, arr=arr16)
    np.save(path, arr=arr16)

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
    out = np.empty((all_tokens, n_layers, hidden_size), dtype=np.float16)

    # 1) Fill in representations for all input tokens from the first step
    for layer_idx in range(n_layers):
        t = data[0][layer_idx]  # shape: [1, n_input_tokens, hidden_size]
        if t.shape[1] != n_input_tokens:
            raise ValueError("Inconsistent seq_len for first new token across layers.")
        out[:n_input_tokens, layer_idx, :] = t[0].detach().cpu().float().numpy().astype(np.float16)

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
            out[current_token_idx, layer_idx, :] = t[0, -1].detach().cpu().float().numpy().astype(np.float16)
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
        dtype=np.float16,
    )

    # Fill with right-aligned attention scores
    for t_idx in range(new_tokens):
        for l_idx in range(n_layers):
            tensor = data[t_idx][l_idx]        # [1, n_heads, 1, seq_len]
            _, _, _, seq_len = tensor.shape

            # Extract attention: [n_heads, seq_len]
            attn = tensor[0, :, -1, :].detach().cpu().float().numpy().astype(np.float16)

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
    out = np.empty((total_tokens, n_layers, activation_dim), dtype=np.float16)

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
            arr = tensor[0].detach().cpu().float().numpy().astype(np.float16)  # (tokens_t, activation_dim)
            out[offset : offset + tokens_t, layer_idx, :] = arr

        offset += tokens_t
    # print('check1====')
    # print(data[0][0][0, 0, :])
    # print(out[0,0,:])
    # print('check2====')
    # print(data[-1][-1][0, 0, :])
    # print(out[-1,-1,:])
    return out