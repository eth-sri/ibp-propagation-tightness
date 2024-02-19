'''
This file should be copied to mn_bab_vnn_2022
Structure:
    - mn_bab_vnn_2022
        - certify.py
    - PI-master
'''
try:
    from src.mn_bab_verifier import MNBaBVerifier
    from src.abstract_layers.abstract_network import AbstractNetwork
    from src.utilities.argument_parsing import get_config_from_json
    from src.utilities.config import make_config
    from src.utilities.loading.network import freeze_network
    from src.concrete_layers.normalize import Normalize as mnbab_normalize
    from src.verification_instance import VerificationInstance
    from src.utilities.initialization import seed_everything as mnbab_seed
    import sys
    sys.path.append("../PI-master/") # add the path of PI-master here
    # We have to do this
    # MN-BaB has some import issues from outside directory
except:
    raise ModuleNotFoundError("There might be an error setting up the file structure of MN-BaB. Please follow the top comment of this .py file.")

import torch
import torch.nn as nn
from args_factory import get_args
from loaders import get_loaders
from PARC_networks import get_network, fuse_BN_wrt_Flatten, remove_BN_wrt_Flatten
from torch_model_wrapper import BoxModelWrapper, BasicModelWrapper
import os
from tqdm import tqdm
import random
import numpy as np
from attacks import adv_whitebox
from AIDomains.zonotope import HybridZonotope
from AIDomains.abstract_layers import Sequential, Flatten
from AIDomains.concrete_layers import Normalization as PARC_normalize
from AIDomains.wrapper import propagate_abs
from AIDomains.wrapper import construct_C
from AIDomains.deeppoly import DeepPoly, backward_deeppoly, forward_deeppoly

from MILP_Encoding.milp_utility import get_layers, verify_network_with_milp, milp_callback
from gurobipy import GRB
import time
from utils import write_perf_to_json, load_perf_from_json, fuse_BN
from MILP_Encoding.milp_utility import build_milp_model
import torch.multiprocessing as mp

import copy

import warnings
warnings.filterwarnings("ignore")



# def block_milp_certify(model_wrapper:BoxModelWrapper, x, y, eps, device, args, timeout=1000, time_relax_step=100, verbose:bool=False, n_class:int=10, bound_type:str="box"):
#     assert bound_type in ["box", "dp_box"], f"Not supported type: {bound_type}"
#     with torch.no_grad():
#         is_milp_verified = torch.zeros(len(x), dtype=torch.bool)
#         is_found_AE = torch.zeros(len(x), dtype=torch.bool)
#         for i in range(len(x)):
#             input = x[i:i+1]
#             target = y[i:i+1]
#             lb = (input - eps).clamp(min=0)
#             ub = (input + eps).clamp(max=1)

#             # manually propagate through the previous blocks
#             model_wrapper.store_box_bounds = True
#             for block in model_wrapper.net_blocks_abs[:-1]:
#                 input = block(input)
#                 lb, ub = model_wrapper.get_IBP_bounds(block, lb, ub)

#             net = model_wrapper.net_blocks_abs[-1]

#             # # use attack to decide if potentially verifiable
#             input_adv = adv_whitebox(net, input, target, lb, ub, device, lossFunc='pgd')
#             target_adv = net(input_adv).argmax(dim=1)
#             if target_adv != target:
#                 is_found_AE[i] = 1
#                 if verbose:
#                     print("Test failed by adv attack.")
#                 continue

#             # certify with MILP for the last block; 
#             # uses DeepPoly and DeepZ to tighten box bound (automatically takes the best bounds)
#             model_wrapper.net.reset_bounds()
#             t1 = time.time()
#             # input_abs = HybridZonotope.construct_from_bounds(lb, ub, domain="zono")
#             # pseudo_bound, pseudo_labels = propagate_abs(net, 'deeppoly', input_abs, target)
#             # abs_out = net(input_abs)
#             input_abs = HybridZonotope.construct_from_bounds(lb, ub, domain="box")
#             # lb1, ub1 = net[4].bounds[0].clone().detach(), net[4].bounds[1].clone().detach()
            
#             # box_lb, box_ub = lb, ub
#             # for layer in net:
#             #     layer.update_bounds((box_lb, box_ub))
#             #     box_lb, box_ub = layer.bounds
#             #     x_abs = HybridZonotope.construct_from_bounds(box_lb, box_ub, domain="box")
#             #     x_abs = layer(x_abs)
#             #     box_lb, box_ub = x_abs.concretize()
#             abs_out = net(input_abs)

#             print(f"Bound Time: {time.time()-t1:.2f}")
#             # lb2, ub2 = net[4].bounds[0].clone().detach(), net[4].bounds[1].clone().detach()
#             # print((lb2 > lb1).sum(), (lb2 < lb1).sum(), (ub2 < ub1).sum(), (ub2 > ub1).sum())
#             # raise
#             t1 = time.time()
#             verified_accu_milp, AE, _ = verify_network_with_milp(net, input_abs, input, target, n_class, partial_milp=-1, max_milp_neurons=-1, timeout=timeout, time_relax_step=time_relax_step, verbose=verbose)
#             if verified_accu_milp:
#                 is_milp_verified[i] = 1
#                 if verbose:
#                     print(f"A new sample verified! Time: {time.time()-t1:.2f}")
#             else:
#                 if AE is not None:
#                     is_found_AE[i] = 1
#                 if verbose:
#                     print(f"An attempt failed. Time: {time.time()-t1:.2f}")

#     return is_milp_verified, is_found_AE




def deeppoly_certify(model_wrapper:BoxModelWrapper, x, y, eps, device, args, verbose:bool=True):
    with torch.no_grad():
        is_dp_certified = torch.zeros(len(x), dtype=torch.bool)
        for i in range(len(x)):
            input = x[i:i+1]
            target = y[i:i+1]
            lb = (input - eps).clamp(min=0)
            ub = (input + eps).clamp(max=1)
            net = model_wrapper.net
            # use attack to decide if potentially verifiable
            input_adv = adv_whitebox(net, input, target, lb, ub, device, lossFunc='pgd')
            target_adv = net(input_adv).argmax(dim=1)
            if target_adv != target:
                if verbose:
                    print("Test failed by adv attack.")
                continue
            x_abs = HybridZonotope.construct_from_bounds(lb, ub, domain="zono")
            pseudo_bound, pseudo_labels = propagate_abs(net, 'deeppoly', x_abs, target)
            if pseudo_bound.argmax(1) == pseudo_labels:
                is_dp_certified[i] = 1
                print("A new sample verified.")
            else:
                print("An attempt failed.")
    return is_dp_certified


def verify_with_mnbab(net, mnbab_verifier, x, y, eps, norm_mean, norm_std, device, mnbab_config, num_classes:int=10):
    is_verified = np.zeros(len(x), dtype=bool)
    is_undecidable = np.zeros(len(x), dtype=bool)
    is_attacked = np.zeros(len(x), dtype=bool)
    for i in range(len(x)):
        net.reset_input_bounds()
        net.reset_output_bounds()
        net.reset_optim_input_bounds()
        input = x[i:i+1]
        label = y[i:i+1]
        input_lb = (input - eps).clamp(min=0)
        input_ub = (input + eps).clamp(max=1)
        # normalize the input here
        input = (input - norm_mean) / norm_std
        input_lb = (input_lb - norm_mean) / norm_std
        input_ub = (input_ub - norm_mean) / norm_std
        with torch.enable_grad():
            inst = VerificationInstance.create_instance_for_batch_ver(net, mnbab_verifier, input, input_lb, input_ub, int(label), mnbab_config, num_classes)
            inst.run_instance()
        if inst.is_verified:
            is_verified[i] = 1
            print("mnbab verifies a new one!")
        if not inst.is_verified and inst.adv_example is None:
            is_undecidable[i] = 1
            print("mnbab cannot decide!")
        if inst.adv_example is not None:
            is_attacked[i] = 1
            print("mnbab finds an adex!")
        inst.free_memory()
    return is_verified, is_undecidable, is_attacked
        

def update_perf(save_root, args, num_cert_ibp, num_nat_accu, num_cert_dp_box, num_mnbab_verified, num_cert_milp, num_total, num_adv_attacked, num_milp_AE, num_milp_timeout, certify_start_time, previous_time, batch_idx, test_loader):
    perf_dict = {
        'block_sizes': args.block_sizes,
        'num_cert_ibp':num_cert_ibp, 
        'num_nat_accu':num_nat_accu, 
        'num_cert_dpb':num_cert_dp_box,
        'num_cert_mnbab':num_mnbab_verified,
        'num_cert_milp':num_cert_milp, 
        'num_total':num_total, 
        'num_adv_attacked':num_adv_attacked,
        "num_milp_AE": num_milp_AE,
        "num_milp_timeout": num_milp_timeout,
        # "num_multi_target_worse": num_multi_target_worse,
        # "num_multi_target_better": num_multi_target_better,
        'nat_accu': num_nat_accu / num_total,
        'ibp_cert_rate': num_cert_ibp / num_total,
        'dpb_cert_rate': num_cert_dp_box / num_total,
        'mnbab_cert_rate': num_mnbab_verified / num_total,
        'adv_unattacked_rate': (num_nat_accu - num_adv_attacked) / num_total,
        'milp_cert_rate': num_cert_milp / num_total,
        "milp_AE_rate": num_milp_AE / num_total,
        "milp_timeout_rate": num_milp_timeout / num_total,
        "total_cert_rate": (num_cert_ibp + num_cert_milp + num_cert_dp_box + num_mnbab_verified) / num_total,
        "total_time": time.time() - certify_start_time + previous_time,
        "batch_remain": len(test_loader) - batch_idx - 1
        }
    write_perf_to_json(perf_dict, save_root, filename="cert.json")
    write_perf_to_json(args.__dict__, save_root, filename="cert_args.json")
    return perf_dict

def transform_abs_into_torch(abs_net, torch_net):
    '''
    load the params in the abs_net into torch net
    '''
    abs_state = abs_net.state_dict()
    torch_state = {}
    for key, value in abs_state.items():
        key = key.lstrip("layers.")
        if key == "0.sigma":
            key = "0.std"
        torch_state[key] = value

    torch_net.load_state_dict(torch_state)
    return torch_net

def switch_normalization_version(torch_net):
    '''
    Using the normalization layer defined in MN-BaB instead
    '''
    for i, layer in enumerate(torch_net):
        if isinstance(layer, PARC_normalize):
            mnbab_layer = mnbab_normalize(layer.mean, layer.std, channel_dim=1)
            torch_net[i] = mnbab_layer
    return torch_net


def run(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config_file = get_config_from_json(args.mnbab_config)
    mnbab_config = make_config(**config_file)

    loaders, num_train, input_size, input_channel, n_class = get_loaders(args, shuffle_test=False) # shuffle test set so that we get reasonable estimation of the statistics even when we don't finish the full test.
    input_dim = (input_channel, input_size, input_size)

    if len(loaders) == 3:
        train_loader, val_loader, test_loader = loaders
    else:
        train_loader, test_loader = loaders
        val_loader = None

    torch_net = get_network(args.net, args.dataset, device)
    torch_net.eval()
    net = Sequential.from_concrete_network(torch_net, input_dim, disconnect=True)
    net.eval()

    if not os.path.isfile(args.load_model):
        raise ValueError(f"There is no such file {args.load_model}.")
    save_root = os.path.dirname(args.load_model)


    net.load_state_dict(torch.load(args.load_model, map_location=device))

    net = fuse_BN_wrt_Flatten(net, device, remove_all=True)
    torch_net = remove_BN_wrt_Flatten(torch_net, device, remove_all=True)
    print(net)


    model_wrapper = BoxModelWrapper(net, nn.CrossEntropyLoss(), (input_channel, input_size, input_size), device, args, block_sizes=args.block_sizes)
    
    model_wrapper.net.eval()

    eps = args.test_eps
    print("Certifying for eps:", eps)
    num_cert_ibp, num_nat_accu, num_cert_dp_box, num_cert_milp, num_total, num_adv_attacked, num_milp_AE, num_milp_timeout, num_mnbab_verified = 0, 0, 0, 0, 0, 0, 0, 0, 0

    previous_time = 0
    if args.load_certify_file:
        # warning: loading from existing file is not always trustable. If a certification is terminated by ctrl-C, the gurobi only shows an info and returns, resulting in a false negative. If the interuption is not too much, then the result should be only very slightly worse.
        perf_dict = load_perf_from_json(save_root, args.load_certify_file)
        if perf_dict is not None:
            num_cert_ibp,  num_nat_accu,  num_cert_dp_box, num_mnbab_verified, num_cert_milp,  num_total,  num_adv_attacked, num_milp_AE, num_milp_timeout, previous_time = perf_dict['num_cert_ibp'], perf_dict['num_nat_accu'], perf_dict['num_cert_dpb'],perf_dict['num_cert_mnbab'],perf_dict['num_cert_milp'], perf_dict['num_total'], perf_dict['num_adv_attacked'],perf_dict['num_milp_AE'],perf_dict['num_milp_timeout'], perf_dict["total_time"]

    model_wrapper.net.set_dim(torch.zeros((test_loader.batch_size, *input_dim), device='cuda'))

    # prepare mn-bab model
    # mnbab use a different normalization class
    torch_net = transform_abs_into_torch(net, torch_net)
    # torch_net = switch_normalization_version(torch_net)
    mnbab_net = AbstractNetwork.from_concrete_module(
        torch_net[1:], mnbab_config.input_dim
    ).to(device) # remove normalization layer, which would be done directly to its input
    freeze_network(mnbab_net)
    mnbab_verifier = MNBaBVerifier(mnbab_net, device, mnbab_config.verifier)


    certify_start_time = time.time()
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            if batch_idx*args.test_batch < num_total:
                continue
            print("Batch id:", batch_idx)
            model_wrapper.net = model_wrapper.net.to(device)
            x, y = x.to(device), y.to(device)
            # 1. try to verify with IBP 
            (loss, nat_loss, cert_loss), (nat_accu, cert_accu), (is_nat_accu, is_cert_accu) = model_wrapper.common_step(x, y, eps, args.test_steps, use_vanilla_ibp=True, summary_accu_stat=False)
            num_nat_accu += is_nat_accu.sum().item()
            num_cert_ibp += is_cert_accu.sum().item()
            num_total += len(x)
            print(f"Batch size: {len(x)}, IBP cert: {is_cert_accu.sum().item()}")

            # only consider classified correct and not IBP verified below
            x = x[is_nat_accu & (~is_cert_accu)]
            y = y[is_nat_accu & (~is_cert_accu)]
            if len(x) == 0:
                perf_dict = update_perf(save_root, args, num_cert_ibp, num_nat_accu, num_cert_dp_box, num_mnbab_verified, num_cert_milp, num_total, num_adv_attacked, num_milp_AE, num_milp_timeout, certify_start_time, previous_time, batch_idx, test_loader)
                continue

            # 2. try to attack with pgd
            is_adv_attacked = torch.zeros(len(x), dtype=torch.bool)
            x_adv = adv_whitebox(model_wrapper.net, x, y, (x-eps).clamp(min=0), (x+eps).clamp(max=1), device, lossFunc='pgd', ODI_num_steps=0, restarts=5, num_steps=args.test_steps)
            y_adv = model_wrapper.net(x_adv).argmax(dim=1)
            is_adv_attacked[(y_adv != y)] = 1
            num_adv_attacked += is_adv_attacked.sum().item()

            # only consider not adv attacked below
            x = x[~is_adv_attacked]
            y = y[~is_adv_attacked]
            if len(x) == 0:
                perf_dict = update_perf(save_root, args, num_cert_ibp, num_nat_accu, num_cert_dp_box, num_mnbab_verified, num_cert_milp, num_total, num_adv_attacked, num_milp_AE, num_milp_timeout, certify_start_time, previous_time, batch_idx, test_loader)
                continue

            # 3. try to verify with dp_box
            data_abs = HybridZonotope.construct_from_noise(x, eps, "box")
            dpb, pesudo_label = propagate_abs(model_wrapper.net, "deeppoly_box", data_abs, y)
            is_dpb_cert = (dpb.argmax(1) == pesudo_label)
            num_cert_dp_box += is_dpb_cert.sum().item()
            x = x[~is_dpb_cert]
            y = y[~is_dpb_cert]
            if len(x) == 0:
                perf_dict = update_perf(save_root, args, num_cert_ibp, num_nat_accu, num_cert_dp_box, num_mnbab_verified, num_cert_milp, num_total, num_adv_attacked, num_milp_AE, num_milp_timeout, certify_start_time, previous_time, batch_idx, test_loader)
                continue

            # 4. try to verify with MN-BaB
            is_verified, is_undecidable, is_mnbab_attacked = verify_with_mnbab(mnbab_net, mnbab_verifier, x, y, eps, torch_net[0].mean, torch_net[0].std, device, mnbab_config, n_class)
            num_mnbab_verified += is_verified.sum().item()
            x = x[is_undecidable]
            y = y[is_undecidable]
            num_adv_attacked += is_mnbab_attacked.sum().item()
            # if len(x) == 0:
            #     perf_dict = update_perf(save_root, args, num_cert_ibp, num_nat_accu, num_cert_dp_box, num_mnbab_verified, num_cert_milp, num_total, num_adv_attacked, num_milp_AE, num_milp_timeout, certify_start_time, batch_idx, test_loader)
            #     continue

            # 4. try to verify with MILP for the last block
            # is_milp_verified, is_found_AE = block_milp_certify(model_wrapper, x, y, eps, device, args, timeout=args.timeout, time_relax_step=args.time_relax_step, verbose=True)
            # is_milp_verified = batch_block_milp_certify(model_wrapper, x, y, eps, device, args, verbose=True)

            # num_cert_milp += is_milp_verified.sum().item()
            # num_milp_AE += is_found_AE.sum().item()
            # num_milp_timeout += ((~is_milp_verified) & (~is_found_AE)).sum().item()

            # # verify with deeppoly
            # is_dp_verified  = deeppoly_certify(model_wrapper, x, y, eps, device, args, verbose=True)

            perf_dict = update_perf(save_root, args, num_cert_ibp, num_nat_accu, num_cert_dp_box, num_mnbab_verified, num_cert_milp, num_total, num_adv_attacked, num_milp_AE, num_milp_timeout, certify_start_time, previous_time, batch_idx, test_loader)


        perf_dict = update_perf(save_root, args, num_cert_ibp, num_nat_accu, num_cert_dp_box, num_mnbab_verified, num_cert_milp, num_total, num_adv_attacked, num_milp_AE, num_milp_timeout, certify_start_time, previous_time, batch_idx, test_loader)
        write_perf_to_json(perf_dict, save_root, filename="complete_cert.json")


        

def main():
    args = get_args()
    run(args)


if __name__ == '__main__':
    mnbab_seed(123)
    main()


