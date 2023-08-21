import torch
import random
import argparse
import os
import numpy as np
import json

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--num_exp', type=int, default=100, help="number of random experiment you want to run.")
    parser.add_argument('--num_layers', type=int, default=12, help='number of TransformerEncoderLayer in upstream model')
    parser.add_argument('--num_paths', type=int, default=4, help='number of candidate adapters in a TransformerEncoderLayer')
    parser.add_argument('--task', type=str, default='sd', help="name of downstream task passing to run_downstream.py by -d '{task_name}'")
    parser.add_argument('--output_root', type=str, default='random_exp/')
    parser.add_argument('--para_budget', type=float, default=0.5, help='maxinum number of parameters in million (M)')

    return parser.parse_args()

num_param = {
    'seq': 0.049952,
    'para': 0.049952,
    'lora': 0.024576,
    'bitfit': 0.008448
}

order = ['seq', 'para', 'lora', 'bitfit']

def get_structure(p, budget):
    curr_num_para = 0.0
    result = [[] for _ in range(len(p))]
    all_p = [(layer_idx, path_idx, value) for layer_idx, path_pair in enumerate(p) for path_idx, value in enumerate(path_pair)]
    all_p = sorted(all_p, key=lambda x:x[2], reverse=True)
    for p_ in all_p:
        layer_idx, path_idx = p_[0], p_[1]
        if num_param[order[path_idx]] + curr_num_para <= budget:
            curr_num_para += num_param[order[path_idx]]
            result[layer_idx].append(order[path_idx])
    
    return result

def custom_sort(result):
    for i in range(len(result)):
        result[i].sort(key=lambda x:order.index(x))
    return result

def main():
    args = get_args()
    output_dir = os.path.join(args.output_root)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'rand.json')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rand_shape = [args.num_layers, args.num_paths]
    results = []
    for i in range(args.num_exp):
        while True:
            rand_p = torch.rand(rand_shape)
            result = get_structure(rand_p, args.para_budget)
            if result not in results:
                results.append(result)
                break
    results = {i: custom_sort(result) for i, result in enumerate(results)}
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    main()