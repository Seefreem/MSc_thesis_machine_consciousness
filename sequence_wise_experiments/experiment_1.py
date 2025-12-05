import json
import os
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
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


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    out_dir = Path(args.output_dir)

    # 1. Load local data
    original_data, ds = load_imdb_sms_for_transformer(args.json_in_path)

    # 3. Load model + tokenizer
    tokenizer, model = setup_model_and_tokenizer(args.model_name)
    print(model)
    device = next(model.parameters()).device
    print('device: ', model.device, device)

    # 4. Register hooks
    mlp_acts, attn_block_outs = register_hooks(model)

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
        
        # Reset hook buffers for this forward pass
        for i in range(len(mlp_acts)):
            mlp_acts[i] = []
            attn_block_outs[i] = []

        # 5. Generate up to MAX_NEW_TOKENS tokens
        with torch.no_grad():
            gen_out = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
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
        base = f"{args.task}_sample_{idx:06d}"

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
    with open(args.json_out_file, "w", encoding="utf-8") as f:
        json.dump(augmented_data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='meta-llama/Llama-3.2-3B-Instruct')
    parser.add_argument("--json_in_path", type=str, default='_datasets/filtered_data/imdb_sms_interval_1_pairs.json')
    parser.add_argument("--output_dir", type=str, default='_datasets/')
    parser.add_argument("--json_out_file", type=str, default='imdb_sms_interval_1_pairs_with_activations.json')
    parser.add_argument("--max_new_tokens", type=int, default=10)
    parser.add_argument("--max_n_samples", type=int, default=None) 
    parser.add_argument(
        "--task", 
        type=str, 
        default='sen_w_t1',
        help='Candedate tasks: sen_w_t1, sen_w_t2, sen_w_b, lay_w_t1, lay_w_t2, lay_w_b, and selective_attention'
        ) 

    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.model_name)
    args.json_out_file = str(args.task) + "_" + args.json_out_file
    args.json_out_file = os.path.join(args.output_dir, args.json_out_file)

    print(f"\n\n ## args: {args} \n\n")
    main(args)
