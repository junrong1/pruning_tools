import argparse
import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM, OPTForCausalLM
from importlib.metadata import version

from lib.eval import eval_ppl

print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())


def get_llm(model, cache_dir="llm_weights"):
    model = LlamaForCausalLM.from_pretrained(
        model,
        torch_dtype=torch.float16,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
        device_map="auto"
    )

    model.seqlen = 2048
    return model


def get_opt(model):
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    model = OPTForCausalLM.from_pretrained(model, torch_dtype='auto')
    model.seqlen = model.config.max_position_embeddings
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='Local Path or Model name')
    parser.add_argument('--model_type', type=str, choices=['opt', 'llm'])
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--sparsity_ratio', type=float, default=0, help='Sparsity level')
    parser.add_argument("--sparsity_type", type=str, choices=["unstructured", "4:8", "2:4"])
    parser.add_argument("--prune_method", type=str, choices=["magnitude", "wanda", "sparsegpt"])
    parser.add_argument("--cache_dir", default="llm_weights", type=str )
    parser.add_argument('--use_variant', action="store_true", help="whether to use the wanda variant described in the appendix")
    parser.add_argument('--save', type=str, default=None, help='Path to save results.')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
    parser.add_argument('--device', type=str, default="cuda", help='device')


    args = parser.parse_args()
    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        assert args.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))

    model = None

    print(f"loading model: {args.model}")
    if args.model_type == 'opt':
        model = get_opt(args.model)
    elif args.model_type == 'llm':
        model = get_llm(args.model)

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    device = torch.device(args.device)
    model.to(device)
    if "30b" in args.model or "65b" in args.model:  # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
        device = model.hf_device_map["lm_head"]
    print("use device ", device)

    if args.model_type == 'opt':
        from lib.prune_opt import prune_wanda, prune_magnitude, prune_sparsegpt, check_sparsity, find_layers
    elif args.model_type == 'llm':
        from lib.prune_llm import prune_wanda, prune_magnitude, prune_sparsegpt, check_sparsity, find_layers

    if args.sparsity_ratio != 0:
        print("pruning starts")
        if args.prune_method == "wanda":
            prune_wanda(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m, model_type=args.model_type)
        elif args.prune_method == "magnitude":
            prune_magnitude(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m, model_type=args.model_type)
        elif args.prune_method == "sparsegpt":
            prune_sparsegpt(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m, model_type=args.model_type)




if __name__ == '__main__':
    main()