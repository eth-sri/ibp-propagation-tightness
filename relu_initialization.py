import torch
import torch.nn as nn
from args_factory import get_args
from loaders import get_loaders
from utils import Scheduler, Statistics
from PARC_networks import get_network, fuse_BN_wrt_Flatten, add_BN_wrt_Flatten
from torch_model_wrapper import BoxModelWrapper, PGDModelWrapper, SmallBoxModelWrapper, GradAccuBoxModelWrapper, GradAccuSmallBoxModelWrapper
import os
from utils import write_perf_to_json, load_perf_from_json, fuse_BN, seed_everything
from tqdm import tqdm
import random
import numpy as np
from regularization import compute_fast_reg, compute_vol_reg, compute_L1_reg, compute_PI_reg, compute_neg_reg
import time
from datetime import datetime
from AIDomains.abstract_layers import Sequential, Flatten, Linear, ReLU, Conv2d

import warnings
warnings.filterwarnings("ignore")

from get_stat import PI_loop, relu_loop, test_loop

def run(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loaders, num_train, input_size, input_channel, n_class = get_loaders(args)
    args.num_classes = n_class
    input_dim = (input_channel, input_size, input_size)
    if len(loaders) == 3:
        train_loader, val_loader, test_loader = loaders
    else:
        train_loader, test_loader = loaders
        val_loader = None
    
    perf_dict = {"relu":{}, "DLN":{}}
    depth_archs = ["cnn_3layer_bn", "cnn_5layer_bn", "cnn_7layer_bn", "cnn_9layer_bn", "cnn_11layer_bn", "cnn_13layer_bn"]
    width_archs = ["cnn_7layer_w32_bn", "cnn_7layer_bn", "cnn_7layer_w128_bn", "cnn_7layer_w256_bn", "cnn_7layer_w512_bn", "cnn_7layer_w1024_bn"]
    # for arch in depth_archs:
    #     net = get_network(arch, args.dataset, device, init=args.init)
    #     net = Sequential.from_concrete_network(net, input_dim)
    #     local_tightness = PI_loop(net, 1e-6, test_loader, device, n_class, args, relu_adjust="local")
    #     DLN_tightness = PI_loop(net, 1e-6, test_loader, device, n_class, args, relu_adjust=None)
    #     perf_dict["relu"][arch] = local_tightness
    #     perf_dict["DLN"][arch] = DLN_tightness
    # write_perf_to_json(perf_dict, "./", "depth_init_tightness.json")

    # theory_archs = ["fc_2layer", ]
    # num_draw = 100
    # for arch in theory_archs:
    #     stats = []
    #     for re in range(num_draw):
    #         net = get_network(arch, args.dataset, device, init=args.init)
    #         net = Sequential.from_concrete_network(net, input_dim)
    #         local_tightness = PI_loop(net, 1e-6, test_loader, device, n_class, args, relu_adjust="local")
    #         DLN_tightness = PI_loop(net, 1e-6, test_loader, device, n_class, args, relu_adjust=None)
    #         stats.append(local_tightness / DLN_tightness)
    #     print(np.mean(stats), np.std(stats)) # close to 1.414


    eps = 1e-2
    perf_dict = {"relu":{}}
    if len(loaders) == 3:
        train_loader, val_loader, test_loader = loaders
    else:
        train_loader, test_loader = loaders
        val_loader = None
    for arch in width_archs:
        net = get_network(arch, args.dataset, device, init=args.init)
        net = Sequential.from_concrete_network(net, input_dim)
        model_wrapper = BoxModelWrapper(net, nn.CrossEntropyLoss(), input_dim, device, args)
        val_nat_accu, val_cert_accu, val_loss = test_loop(model_wrapper, eps, val_loader if val_loader is not None else test_loader, device, args)
        perf_dict["relu"][arch] = val_loss
    write_perf_to_json(perf_dict, "./", f"width_{args.init}_init_IBP_loss.json")

def main():
    args = get_args()
    seed_everything(args.random_seed)
    run(args)

if __name__ == '__main__':
    main()

