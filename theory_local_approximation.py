import torch
import torch.nn as nn
from args_factory import get_args
from loaders import get_loaders
from utils import Scheduler, Statistics
from PARC_networks import get_network, fuse_BN_wrt_Flatten, add_BN_wrt_Flatten
from torch_model_wrapper import BoxModelWrapper, SmallBoxModelWrapper
import os
from utils import write_perf_to_json, load_perf_from_json, fuse_BN
from tqdm import tqdm
import random
import numpy as np
from regularization import compute_fast_reg, compute_vol_reg, compute_L1_reg
import time
from datetime import datetime
from AIDomains.abstract_layers import Sequential, Flatten, Linear, ReLU, Conv2d
from AIDomains.wrapper import propagate_abs
from AIDomains.zonotope import HybridZonotope
from AIDomains.ai_util import construct_C
from attacks import adv_whitebox

from MILP_Encoding.milp_utility import get_bound_with_milp
from PI_functions import compute_tightness

def run(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    perf_dict = {}
    perf_dict["start_time"] = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    verbose = False

    loaders, num_train, input_size, input_channel, n_class = get_loaders(args, shuffle_test=False)
    input_dim = (input_channel, input_size, input_size)
    if len(loaders) == 3:
        train_loader, val_loader, test_loader = loaders
    else:
        train_loader, test_loader = loaders
        val_loader = None

    net = get_network(args.net, args.dataset, device)

    net = Sequential.from_concrete_network(net, input_dim, disconnect=True)
    print(net)

    if args.load_model:
        net.load_state_dict(torch.load(args.load_model))


    os.makedirs(args.save_dir, exist_ok=True)
    trunc = 500

    eps = args.test_eps
    all_bounds = {"true_PI":[], "local_PI":[]}
    with torch.no_grad():
        for idx, (x, y) in enumerate(tqdm(test_loader)):
            if idx == trunc:
                break
            x = x.to(device)
            y = y.to(device)
            for i in range(len(x)):
                input = x[i:i+1]
                target = y[i:i+1]

                abs_input = HybridZonotope.construct_from_noise(input, eps, domain="box")
                C = construct_C(n_class, target)
                abs_out = net(abs_input, C=C)
                lb, ub = abs_out.concretize()
                IBP_size = (ub - lb).cpu()

                try:
                    MILP_lb = - get_bound_with_milp(net, abs_input, input, target, n_class, verbose=False, mode="lower")
                    MILP_ub = - get_bound_with_milp(net, abs_input, input, target, n_class, verbose=False, mode="upper")
                    exact_size = MILP_ub - MILP_lb
                    true_PI = (exact_size.clamp(min=1e-8) / IBP_size.clamp(min=1e-8))
                except:
                    print("MILP failed; skipping this example.")
                    continue

                local_PI = compute_tightness(net, input, target, eps, num_classes=n_class, relu_adjust="local").cpu()

                all_bounds["true_PI"].append(true_PI)
                all_bounds["local_PI"].append(local_PI)


        for key in all_bounds.keys():
            all_bounds[key] = np.concatenate(all_bounds[key])
            np.save(f"{os.path.join(args.save_dir, key)}_{eps}", all_bounds[key])

def main():
    args = get_args()
    run(args)

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    seed_everything(123)
    main()
