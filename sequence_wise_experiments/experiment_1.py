import json
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from load_datasets import load_imdb_sms_for_transformer

JSON_IN_PATH = "_datasets/filtered_data/imdb_sms_interval_1_pairs.json"
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
OUTPUT_DIR = "_datasets/" + MODEL_NAME
JSON_OUT_PATH = OUTPUT_DIR + "/imdb_sms_interval_1_pairs_with_llama_activations.json"
MAX_NEW_TOKENS = 10
MAX_SAMPLES = 1  # set to an int to debug on a subset


PROMPT_TEMPLATE = (
    "context 1: {c1}; \ncontext 2: {c2}.\n "
    "Task: sentiment analysis of context 1, positive or negative.\n"
    "Sentiment of context 1 is _"
    # "Among the labels of positive and negative, the sentiment label of context 1 is: "
)


def wrap_example(context_1: str, context_2: str) -> str:
    """Create the prompt string for one example."""
    return PROMPT_TEMPLATE.format(c1=context_1, c2=context_2)


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

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_dir = Path(OUTPUT_DIR)

    # 1. Load local data
    original_data, ds = load_imdb_sms_for_transformer(JSON_IN_PATH)

    # 3. Load model + tokenizer
    tokenizer, model = setup_model_and_tokenizer(MODEL_NAME)
    print(model)
    device = next(model.parameters()).device # same
    print('device: ', model.device, device)

    # 4. Register hooks
    mlp_acts, attn_block_outs = register_hooks(model)

    # The data updated with the activations and model generations
    augmented_data = []
    num_examples = len(ds) if MAX_SAMPLES is None else min(len(ds), MAX_SAMPLES)

    for idx in tqdm(range(num_examples), desc="Processing examples"):
        example = ds[idx]
        orig_example = original_data[idx]

        # 2. Wrap data into prompt
        prompt = wrap_example(example["context_1"], example["context_2"])
        # print('prompt:\n', type(prompt), prompt)

        # Tokenize
        enc = tokenizer(
            [prompt],
            return_tensors="pt",
            padding=False,
            truncation=True,
        ).to(device)
        # print(enc.keys(), type(enc['input_ids']), len(enc['input_ids'][0]))
        
        # Reset hook buffers for this forward pass
        for i in range(len(mlp_acts)):
            mlp_acts[i] = []
            attn_block_outs[i] = []

        # 5. Generate up to MAX_NEW_TOKENS tokens
        with torch.no_grad():
            gen_out = model.generate(
                **enc,
                max_new_tokens=MAX_NEW_TOKENS,
                return_dict_in_generate=True,
                output_scores = True,
                output_logits = True,
                output_hidden_states=True,
                output_attentions = True,
                use_cache = True,
                return_legacy_cache=True,
                do_sample=False,
            )

        # for g in gen_out['sequences']:
        #     print(type(g), len(g), g.shape, flush=True)
        #     # <class 'torch.Tensor'> 234 torch.Size([234])

        full_text = tokenizer.decode(
            gen_out['sequences'][0],
            skip_special_tokens=True
        )
        generated_text = tokenizer.decode(
            gen_out['sequences'][0][enc["input_ids"].shape[-1]:]
        )
        # print('generated_text: ', generated_text)
        # print('Length of new tokens:', len(gen_out['sequences'][0][enc["input_ids"].shape[-1]:]))

        # hidden_states is a tuple of (new_tokens, n_layers, [BS, 1 or n_in_tokes, hidden_size])
        # output nd.array.shape: [all_tokens, n_layers, hidden_size]
        hidden_states_np = tuple_to_numpy_all_tokens(gen_out['hidden_states'])
        # for go in gen_out['hidden_states']:
        #     print(go[0].shape)
        print('hidden_states_np.shape', hidden_states_np.shape)
        # Attention scores (per layer)
        # new_tokens x (layers) x [BS, heads, 1, whole_sequence]
        # since the length of attention vectors increase during generation, we use zero-left-padding in ndarray
        # output nd.array.shape: (new_tokens, n_layers, n_heads, max_seq_len)
        attention_scores_np = attn_tuple_to_padded_numpy(gen_out['attentions'])
        print('attention_scores_np.shape', attention_scores_np.shape)
        # MLP activations (per layer, from hooks)
        # mlp_acts_np = tensor_list_to_numpy(mlp_acts)  # [num_layers, seq_len, hidden_dim]
        mlp_acts_np = layer_token_tuple_to_numpy(mlp_acts)
        print('mlp_acts_np.shape', mlp_acts_np.shape)
        # print('mlp_acts:', len(mlp_acts), len(mlp_acts[0]))
        # for i2 in mlp_acts[0]:
        #     print(i2.shape)
        
        # # Attention block outputs (combined multi-head output after o_proj)
        # attn_block_outs_np = tensor_list_to_numpy(attn_block_outs)  # [num_layers, seq_len, hidden_dim]
        attn_block_outs_np = layer_token_tuple_to_numpy(attn_block_outs)
        print('attn_block_outs_np.shape', attn_block_outs_np.shape)
        # print('attn_block_outs', len(attn_block_outs), len(attn_block_outs[0]))
        # for i2 in attn_block_outs[0]:
        #     print(i2.shape)

        # Final logits and probabilities for each *new* token
        logits_new = torch.cat(gen_out['logits'], 0).cpu().float().numpy().astype(np.float16)  #  x [new_tokens, vocab_size]
        scores_new = torch.cat(gen_out['scores'], 0).cpu().float().numpy().astype(np.float16)  # x [new_tokens, vocab_size]
        # print('logits:\n',gen_out['logits'])
        # print('logits_new:\n',logits_new)
        # print('scores:\n',gen_out['scores'])
        # print('scores_new:\n',scores_new)
        # logits_new_np = logits_new.numpy()
        print('logits_new.shape', logits_new.shape)
        print('scores_new.shape', scores_new.shape)

        # 6. Save arrays to .npy files
        base = f"sample_{idx:06d}"

        hidden_file = out_dir / f"{base}_hidden_states.npy"
        attn_file = out_dir / f"{base}_attentions.npy"
        mlp_file = out_dir / f"{base}_mlp_activations.npy"
        attn_out_file = out_dir / f"{base}_attn_block_outputs.npy"
        logits_file = out_dir / f"{base}_logits_new.npy"
        probs_file = out_dir / f"{base}_scores_new.npy"

        np.save(hidden_file, hidden_states_np)
        np.save(attn_file, attention_scores_np)
        np.save(mlp_file, mlp_acts_np)
        np.save(attn_out_file, attn_block_outs_np)
        np.save(logits_file, logits_new)
        np.save(probs_file, scores_new)

        # 7. Add file names + generated text into original JSON object
        new_obj = dict(orig_example)
        new_obj.update(
            {
                "wrapped_prompt": prompt,
                "generated_text": generated_text,
                "hidden_states_file": str(hidden_file),
                "attentions_file": str(attn_file),
                "mlp_activations_file": str(mlp_file),
                "attn_block_outputs_file": str(attn_out_file),
                "logits_new_file": str(logits_file),
                "scores_new_file": str(probs_file),
            }
        )
        augmented_data.append(new_obj)

    # Save updated JSON list
    with open(JSON_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(augmented_data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
