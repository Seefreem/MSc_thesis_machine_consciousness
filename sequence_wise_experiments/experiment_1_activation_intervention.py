import json
import os
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
import joblib

import numpy as np
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM 
from tqdm import tqdm
from load_datasets import load_imdb_sms_for_transformer
from myutilities import tuple_to_numpy_all_tokens
from myutilities import attn_tuple_to_padded_numpy
from myutilities import layer_token_tuple_to_numpy
from myutilities import register_hooks
from myutilities import setup_model_and_tokenizer

def wrap_example(context_1: str, context_2: str, template_type: str) -> str:
    """Create the prompt string for one example."""
    PROMPT_TEMPLATE = ""
    if template_type == 'sen_w_t1':
        PROMPT_TEMPLATE = (
            "context 1: {c1}; \ncontext 2: {c2}.\n "
            "Task: sentiment analysis of context 1, positive or negative.\n"
            "Sentiment of context 1 is _"
            # "Among the labels of positive and negative, the sentiment label of context 1 is: "
        )
        PROMPT_TEMPLATE = PROMPT_TEMPLATE.format(c1=context_1, c2=context_2)
    elif template_type == 'sen_w_t2':
        PROMPT_TEMPLATE = (
            "context 1: {c1}; \ncontext 2: {c2}.\n "
            "Task: classify SMS incontext 2 as spam or ham (not a spam).\n"
            "Context 2 is classified as _"
        )
        PROMPT_TEMPLATE = PROMPT_TEMPLATE.format(c1=context_1, c2=context_2)
    elif template_type == 'sen_w_b':
        PROMPT_TEMPLATE = (
            "context 1: {c1}; \ncontext 2: {c2}.\n "
            # "Task: classify SMS incontext 2 as spam or ham (not a spam).\n"
            # "Context 2 is classified as _"
        )
        PROMPT_TEMPLATE = PROMPT_TEMPLATE.format(c1=context_1, c2=context_2)
    elif template_type == 'lay_w_t1':
        raise ValueError(f"Unhandled template type {template_type}")
    elif template_type == 'lay_w_t2':
        raise ValueError(f"Unhandled template type {template_type}")
    elif template_type == 'lay_w_b':
        raise ValueError(f"Unhandled template type {template_type}")
    elif template_type == 'selective_attention':
        raise ValueError(f"Unhandled template type {template_type}")
    else:
        raise ValueError(f"Unknown template type {template_type}")
    return PROMPT_TEMPLATE

def register_hooks(args, model, layer_idx, alpha=1):
    """
    Rep′ = Rep - α p Prob, α∈[-1, 1]
    """
    # Load probe for fold 0  HERE
    base_dir = os.path.join(args.probe_base_dir, args.model_name, args.task)
    probe_dir = os.path.join(
        base_dir, "linear_probes_logreg_inf", f"layer_{layer_idx}"
    )
    probe_path = os.path.join(
        probe_dir, f"logreg_layer{layer_idx}_fold0.joblib"
    )
    print(f"  Loading probe from {probe_path}")
    # bundle = joblib.load(probe_path)
    probe = joblib.load(probe_path)
    clf = probe["classifier"]
    w = clf.coef_.reshape(-1)
    w_norm = np.linalg.norm(w) + 1e-9
    w_unit = w / w_norm
    direction = torch.from_numpy(w_unit).float()

    def hook_fn(module, inputs, output):
        """
        output: expected shape (batch, seq_len, hidden_dim)
                or a tuple whose first element is that tensor.
        We edit all tokens' representations.
        """
        # Handle cases where HF returns (hidden_states, ...)
        if isinstance(output, tuple):
            hidden = output[0]
            other = output[1:]
        else:
            hidden = output
            other = None

        # hidden: (B, T, D)
        B, T, D = hidden.shape

        # Move direction to correct device / dtype
        dir_vec = direction.to(hidden.device, dtype=hidden.dtype)  # (D,)
        dir_vec = dir_vec.view(1, 1, D)  # broadcast over (B, T, D)

        # Compute scalar projection p for each token: (B, T, 1)
        # p = <Rep, Prob>
        p = (hidden * dir_vec).sum(dim=-1, keepdim=True)

        # Rep' = Rep - alpha * p * Prob
        edited_hidden = hidden - alpha * p * dir_vec

        if other is None:
            return edited_hidden
        else:
            # Repack into the original structure
            return (edited_hidden,) + other

    # Llama-style: model.model.layers is a list of decoder layers
    decoder_layers = model.model.layers
    num_layers = len(decoder_layers)
    print(f"num_layers: {num_layers}")
    if isinstance(model, LlamaForCausalLM):
        # NOTE, that the number of layers is 1 less than the stored hidden representation layers
        decoder_layers[layer_idx - 1].register_forward_hook(hook_fn) 
    else:
        raise ValueError(f'Not handled model type {type(model)}')

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    out_dir = Path(args.output_dir)

    # 1. Load local data
    original_data, ds = load_imdb_sms_for_transformer(args.json_in_path)

    # 3. Load model + tokenizer
    tokenizer, model = setup_model_and_tokenizer(args.model_name)
    # tokenizer, model = None, None
    print(model)
    # 4. Register hooks
    # mlp_acts, attn_block_outs = register_hooks(model)
    register_hooks(args, model, layer_idx=args.layer_idx, alpha=args.alpha)

    device = next(model.parameters()).device
    print('device: ', model.device, device)

    # The data updated with the activations and model generations
    augmented_data = []
    num_examples = len(ds) if args.max_n_samples is None else min(len(ds), args.max_n_samples)

    for idx in tqdm(range(num_examples), desc="Processing examples"):
        example = ds[idx]
        orig_example = original_data[idx]

        # 2. Wrap data into prompt
        prompt = wrap_example(example["context_1"], example["context_2"], args.task)
        # print('prompt:\n', type(prompt), prompt)

        # Tokenize
        enc = tokenizer(
            [prompt],
            return_tensors="pt",
            padding=False,
            truncation=True,
        ).to(device)
        # print(enc.keys(), type(enc['input_ids']), len(enc['input_ids'][0]))

        # 5. Generate up to MAX_NEW_TOKENS tokens
        with torch.no_grad():
            gen_out = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                return_dict_in_generate=True,
                output_logits = True,
                output_hidden_states=True,
                # output_attentions = True,
                use_cache = True,
                return_legacy_cache=True,
                do_sample=False,
            )

        generated_text = tokenizer.decode(
            gen_out['sequences'][0][enc["input_ids"].shape[-1]:]
        )
        hidden_states_np = tuple_to_numpy_all_tokens(gen_out['hidden_states'])
        print('hidden_states_np.shape', hidden_states_np.shape)

        # Final logits and probabilities for each *new* token
        logits_new = torch.cat(gen_out['logits'], 0).cpu().float().numpy().astype(np.float16)  #  x [new_tokens, vocab_size]

        print('logits_new.shape', logits_new.shape)

        # 6. Save arrays to .npy files
        base = f"{args.task}_sample_{idx:06d}"

        hidden_file = out_dir / f"{base}_hidden_states.npy"
        logits_file = out_dir / f"{base}_logits_new.npy"

        np.save(hidden_file, hidden_states_np)
        np.save(logits_file, logits_new)

        # 7. Add file names + generated text into original JSON object
        new_obj = dict(orig_example)
        new_obj.update(
            {
                "wrapped_prompt": prompt,
                "generated_text": generated_text,
                "hidden_states_file": str(hidden_file),
                "logits_new_file": str(logits_file),
            }
        )
        augmented_data.append(new_obj)

    # Save updated JSON list
    with open(args.json_out_file, "w", encoding="utf-8") as f:
        json.dump(augmented_data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='meta-llama/Llama-3.1-8B-Instruct')
    parser.add_argument("--json_in_path", type=str, default='_datasets/filtered_data/imdb_sms_interval_1_pairs.json')
    parser.add_argument("--output_dir", type=str, default='_datasets')
    parser.add_argument("--json_out_file", type=str, default='imdb_sms_interval_1_pairs_with_activations.json')
    parser.add_argument("--max_new_tokens", type=int, default=10)
    parser.add_argument("--probe_base_dir", type=str, default='_datasets/HPC')
    parser.add_argument("--max_n_samples", type=int, default=None) 
    parser.add_argument("--alpha", type=float, default=1.0, 
        help='Alpha is a scalar. When alpha is 1.0, it removes the component on the direction of the probe; '
             'when alpha is -1, it doubles the component on the direction of the probe. ') 
    parser.add_argument(
        "--layer_idx",
        type=int,
        default=None,
        required=True,
        help='The layer to modify (e.g. "0", "5")',
    )
    parser.add_argument(
        "--task", 
        type=str, 
        default='sen_w_t1',
        help='Candidate tasks: sen_w_t1, sen_w_t2, sen_w_b, lay_w_t1, lay_w_t2, lay_w_b, and selective_attention'
        ) 

    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, 'activation_inter', 
        args.model_name, args.task, str(args.layer_idx))
    args.json_out_file = str(args.task) + "_" + args.json_out_file
    args.json_out_file = os.path.join(args.output_dir, args.json_out_file)

    print(f"\n\n ## args: {args} \n\n")
    main(args)
