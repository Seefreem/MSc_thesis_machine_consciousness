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
from load_datasets import load_iab_for_transformer
from myutilities import tuple_to_numpy_all_tokens
from myutilities import attn_tuple_to_padded_numpy
from myutilities import layer_token_tuple_to_numpy
from myutilities import register_hooks
from myutilities import setup_model_and_tokenizer
from transformers.cache_utils import DynamicCache

@torch.inference_mode()
def generate_with_return_cache(
    model, 
    input_ids, 
    past_key_values,
    max_new_tokens=10, 
    do_sample=False):
    """
    Returns:
      - full sequences (prompt + generated)
      - KV cache: iterable <keys, values> pairs
    """
    if isinstance(model.generation_config.eos_token_id, list):
        pad_token_id = model.generation_config.eos_token_id[0]
    else:
        pad_token_id = model.generation_config.eos_token_id
    
    out = model.generate(
        **input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        return_dict_in_generate=True,
        use_cache=True,
        past_key_values=past_key_values,
        pad_token_id=pad_token_id,
        eos_token_id=model.generation_config.eos_token_id,
    )
    # for keys, values in past_key_values:
    #     print(keys.shape, values.shape) # [batch_size, num_heads, seq_len, head_dim]
    return out.sequences, past_key_values

def normal_kv_cache_mode(model, tokenizer, first_prompt, 
                         example, past_key_values_flag=True, device=None, verbose=False):
    messages = []
    messages.append({"role": "user", "content": first_prompt})
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, 
                                              return_tensors="pt", return_dict=True).to(device)
    input_length = input_ids["input_ids"].shape[1]
    # input_ids = tokenizer(first_prompt, return_tensors="pt", 
    #                       padding=False, truncation=True).to(device)
        
    # First question
    past_key_values = DynamicCache()
    full_seq, kv_cache = generate_with_return_cache(
        model, input_ids, past_key_values, max_new_tokens=256)
    # full_seq = tokenizer.decode(full_seq[0])
    completion = tokenizer.decode(full_seq[0, input_length: ], skip_special_tokens=True)
    messages.append({"role": "assistant", "content": completion})
    if verbose:
        print('generated text:', completion)
    # print(type(kv_cache))

    # Second question
    # prompt_2 = full_seq + ".\n" + example['question_2']
    # if 'meta' in args.model_name:
    #     prompt_2 += '_'
    # elif 'google' in args.model_name:
    #     prompt_2 += '**'
    messages.append({"role": "user", "content": example['question_2']})
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, 
                                              return_tensors="pt", return_dict=True).to(device)
    
    # input_ids = tokenizer(prompt_2, return_tensors="pt", 
    #                         padding=False, truncation=True).to(device)
    if past_key_values_flag: ## kv-cache mode
        full_seq_rcp, kv_cache_rcp = generate_with_return_cache(
            model, input_ids, past_key_values=past_key_values)
    else: ## recompute mode
        full_seq_rcp, kv_cache_rcp = generate_with_return_cache(
            model, input_ids, past_key_values=None)
    full_seq_rcp = tokenizer.decode(full_seq_rcp[0])
    return full_seq_rcp, kv_cache_rcp

# masked kv-cache mode
def masked_kv_cache_mode(model, tokenizer, first_prompt, example, device, verbose=False):
    # input_ids = tokenizer(first_prompt, return_tensors="pt", 
    #                         padding=False, truncation=True).to(device)
    messages = []
    messages.append({"role": "user", "content": first_prompt})
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, 
                                              return_tensors="pt", return_dict=True).to(device)
    input_length = input_ids["input_ids"].shape[1]
    
    past_key_values = DynamicCache()
    full_seq, kv_cache = generate_with_return_cache(
        model, input_ids, past_key_values, max_new_tokens=256)
    # full_seq = tokenizer.decode(full_seq[0])
    # prompt_2 = full_seq + ".\n" + example['question_2']
    completion = tokenizer.decode(full_seq[0, input_length: ], skip_special_tokens=True)
    messages.append({"role": "assistant", "content": completion})
    if verbose:
        print('generated text:', completion)

    messages.append({"role": "user", "content": example['question_2']})
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, 
                                              return_tensors="pt", return_dict=True)

    # if 'meta' in args.model_name:
    #     prompt_2 += '_'
    # elif 'google' in args.model_name:
    #     prompt_2 += '**'

    # input_ids = tokenizer(prompt_2, return_tensors="pt", 
    #                         padding=False, truncation=True)
    context_ids = tokenizer(first_prompt, add_special_tokens=False, 
                            return_tensors="pt")
    len_span = len(context_ids['input_ids'][0])
    input_length = len(input_ids['input_ids'][0])
    if verbose:
        print('======input_length', input_length)
        print('input_length, len_span', input_length, len_span)
        print('context:\n', tokenizer.convert_ids_to_tokens(context_ids["input_ids"][0]))
    
    for i in range(input_length - len_span):
        if torch.all(context_ids['input_ids'][0] == input_ids["input_ids"][0][i:i + len_span]):
            input_ids["attention_mask"][0][i:i + len_span - 1] *= 0
            if verbose:
                print('Found the span, idx starts with', i)
                # Allow the model to attend to the last token of the input.
                print("**Masked tokens**:", tokenizer.decode(input_ids["input_ids"][0][i:i + len_span - 1]))
            break
    if verbose:
        print('attention_mask:', input_ids["attention_mask"])
    
    input_ids.to(device)
    full_seq_2, kv_cache_2 = generate_with_return_cache(
        model, input_ids, past_key_values)
    
    full_seq_2 = tokenizer.decode(full_seq_2[0])
    return full_seq_2, kv_cache_2

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    out_dir = Path(args.output_dir)

    # 1. Load local data
    original_data, ds = load_iab_for_transformer(args.json_in_path)

    # 3. Load model + tokenizer
    tokenizer, model = setup_model_and_tokenizer(args.model_name)
    print(model)
    device = next(model.parameters()).device
    print('device: ', model.device, device)
    n_verbose = 5
    for exp_idx, example in tqdm(enumerate(ds)):
        # print(f'===============Sample index {exp_idx}=================')
        prompt_template = example['prompt_template']
        if 'meta' in args.model_name:
            prompt_template += '_'
        elif 'google' in args.model_name:
            prompt_template += '**'
        first_prompt = prompt_template.format(c1=example['question_1'], 
                                                 c2=example['activities'])
        # print()
        # print(first_prompt)
        # Recompute mode
        if exp_idx < n_verbose:
            print('===========recompute mode:\n')
        full_seq_rcp, kv_cache_rcp = normal_kv_cache_mode(
            model, tokenizer, first_prompt, example, 
            past_key_values_flag=False, device=device,
            verbose=(n_verbose >= exp_idx))
        original_data[exp_idx]['final_seq_recomputed'] = str(full_seq_rcp)
        if exp_idx < n_verbose:
            print('teacher:',  example['teacher'])
            print('cleaner:',  example['cleaner'])
            print('dog:',      example['dog'])
            print('homework:', example['homework'])
            print('Final test:', full_seq_rcp)
        
        ## KV-cache mode
        if exp_idx < n_verbose:
            print('===========KV-cache mode:\n')
        full_seq_kv, kv_cache_kv = normal_kv_cache_mode(
            model, tokenizer, first_prompt, example, 
            past_key_values_flag=True, device=device,
            verbose=(n_verbose >= exp_idx))
        original_data[exp_idx]['final_seq_kv_cache'] = str(full_seq_kv)
        if exp_idx < n_verbose:
            print('teacher:',  example['teacher'])
            print('cleaner:',  example['cleaner'])
            print('dog:',      example['dog'])
            print('homework:', example['homework'])
            print('Final text:', full_seq_kv)

        ## masked kv-cache mode
        if exp_idx < n_verbose:
            print('===========masked kv-cache mode:\n')
        full_seq_mkv, kv_cache_mkv = masked_kv_cache_mode(
            model, tokenizer, first_prompt, example, device,
            verbose=(n_verbose >= exp_idx))
        original_data[exp_idx]['final_seq_masked_kv'] = str(full_seq_mkv)
        if exp_idx < n_verbose:
            print('teacher:',  example['teacher'])
            print('cleaner:',  example['cleaner'])
            print('dog:',      example['dog'])
            print('homework:', example['homework'])
            print('Final text:', full_seq_mkv)
        
        if exp_idx == 20:
            break
    # Save updated JSON list
    with open(args.json_out_file, "w", encoding="utf-8") as f:
        json.dump(original_data, f, ensure_ascii=False, indent=2)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='meta-llama/Llama-3.2-3B-Instruct')
    parser.add_argument("--json_in_path", type=str, default='_datasets/IAB/iab.json')
    parser.add_argument("--output_dir", type=str, default='_datasets/')
    parser.add_argument("--json_out_file", type=str, default='data_with_activations.json')
    parser.add_argument("--max_new_tokens", type=int, default=10)
    parser.add_argument("--max_n_samples", type=int, default=None) 
    parser.add_argument(
        "--task", 
        type=str, 
        default='selective_attention',
        help=('selective_attention')
        ) 

    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.model_name, args.task)
    args.json_out_file = str(args.task) + "_" + args.json_out_file
    args.json_out_file = os.path.join(args.output_dir, args.json_out_file)

    print(f"\n\n ## args: {args} \n\n")
    main(args)
