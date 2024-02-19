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
from regularization import compute_fast_reg, compute_vol_reg, compute_L1_reg
import time
from datetime import datetime
from AIDomains.abstract_layers import Sequential, Flatten, Linear, ReLU, Conv2d, _BatchNorm
from AIDomains.zonotope import HybridZonotope
from AIDomains.ai_util import construct_C

from PI_functions import compute_tightness

import warnings
warnings.filterwarnings("ignore")

def test_loop(model_wrapper, eps, test_loader, device, args):
    model_wrapper.net.eval()
    model_wrapper.store_box_bounds = False
    nat_accu_stat, cert_accu_stat, loss_stat = Statistics.get_statistics(3)

    use_vanilla_ibp = args.use_vanilla_ibp

    pbar = tqdm(test_loader)
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)
            (loss, nat_loss, cert_loss), (nat_accu, cert_accu) = model_wrapper.common_step(x, y, eps, args.test_steps, use_vanilla_ibp) # already called eval, so do not need to close BN again in the common step.
            nat_accu_stat.update(nat_accu, len(x))
            cert_accu_stat.update(cert_accu, len(x))
            loss_stat.update(loss.item(), len(x))
            pbar.set_postfix_str(f"nat_accu: {nat_accu_stat.avg:.3f}, cert_accu: {cert_accu_stat.avg:.3f}")
    return nat_accu_stat.avg, cert_accu_stat.avg, loss_stat.avg


def PI_loop(net, eps, test_loader, device, num_classes, args, relu_adjust="local"):
    net.eval()
    BN_layers = [layer for layer in net if isinstance(layer, _BatchNorm)]
    for layer in BN_layers:
        layer.set_current_to_running() # Essential for testing; compute_tightness will use current stat for computation
    # fused_net = fuse_BN_wrt_Flatten(net, device, remove_all=True)


    vanilla_PI_stat, local_PI_stat, shrink_PI_stat, center_PI_stat = Statistics.get_statistics(4)

    pbar = tqdm(test_loader)
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)
            # vanilla = compute_tightness(net, x, y, eps, num_classes=num_classes, relu_adjust=None)
            # vanilla_PI_stat.update(vanilla.mean().item(), len(x)) 
            # shrink = compute_tightness(net, x, y, eps, num_classes=num_classes, relu_adjust="shrink")
            # shrink_PI_stat.update(shrink.mean().item(), len(x))
            # print("Fused:")
            # local = compute_tightness(fused_net, x, y, eps, num_classes=num_classes, relu_adjust="local", error_check=True)

            # print("Original:")
            local = compute_tightness(net, x, y, eps, num_classes=num_classes, relu_adjust=relu_adjust, error_check=False)
            local_PI_stat.update(local.mean().item(), len(x))
            # center = compute_tightness(net, x, y, eps, num_classes=num_classes, relu_adjust="center")
            # center_PI_stat.update(center.mean().item(), len(x))

            # pbar.set_postfix_str(f"vanilla_PI: {vanilla_PI_stat.avg:.3f}, local_PI: {local_PI_stat.avg:.3f}, shrink_PI: {shrink_PI_stat.avg:.3f}, center_PI: {center_PI_stat.avg:.3f}")
            pbar.set_postfix_str(f"local_PI: {local_PI_stat.avg:.3f}")

    # return vanilla_PI_stat.avg, local_PI_stat.avg, shrink_PI_stat.avg, center_PI_stat.avg
    net.reset_bounds()
    return local_PI_stat.avg

def relu_loop(net, eps, test_loader, device, args):
    net.eval()
    BN_layers = [layer for layer in net if isinstance(layer, _BatchNorm)]
    relu_layers = [layer for layer in net if isinstance(layer, ReLU)]

    original_stat = [layer.update_stat for layer in BN_layers]
    for layer in BN_layers:
        layer.update_stat = False

    dead, unstable, active = Statistics.get_statistics(3)
    pbar = tqdm(test_loader)
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(pbar):
            x = x.to(device)
            num_dead, num_active, num_total = 0, 0, 0
            net.reset_bounds()
            abs_input = HybridZonotope.construct_from_noise(x, eps, "box")
            abs_out = net(abs_input)
            for layer in relu_layers:
                lb, ub = layer.bounds
                num_total += lb.numel()
                num_dead += (ub < 0).sum().item()
                num_active += (lb > 0).sum().item()
            num_unstable = num_total - num_dead - num_active
            dead.update(num_dead/num_total, len(x))
            unstable.update(num_unstable/num_total, len(x))
            active.update(num_active/num_total, len(x))
            pbar.set_postfix_str(f"dead: {dead.avg:.3f}; unstable: {unstable.avg:.3f}; active: {active.avg:.3f}")

    
    for layer, stat in zip(BN_layers, original_stat):
        layer.update_stat = stat
    net.reset_bounds()
    return dead.avg, unstable.avg, active.avg

def BoxSize_Loop(net, eps, test_loader, device, num_class:int, args):
    net.eval()
    bs_stat = Statistics()
    net.reset_bounds()
    pbar = tqdm(test_loader)
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)
            abs_input = HybridZonotope.construct_from_noise(x, eps, "box")
            C = construct_C(num_class, y)
            abs_out = net(abs_input, C=C)
            lb, ub = abs_out.concretize()
            bs = ((ub - lb) / 2).mean()
            bs_stat.update(bs.item(), len(x))
            net.reset_bounds()
            pbar.set_postfix_str(f"Box_size: {bs_stat.avg:.3E}")
    return bs_stat.avg

def Margin_Loop(net, test_loader, device, num_class:int, args):
    # Computes the margin for the natural inputs. Margin is defined as largest logit minus the second largest logit
    net.eval()
    margin_stat = Statistics()
    net.reset_bounds()
    pbar = tqdm(test_loader)
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)
            out = net(x)
            top2, _ = torch.topk(out, k=2, dim=1)
            margin = (top2[:, 0] - top2[:, 1]).abs().mean()
            margin_stat.update(margin.item(), len(x))
            net.reset_bounds()
            pbar.set_postfix_str(f"Margin: {margin_stat.avg:.3E}")
    return margin_stat.avg

def run_PI(args, relu_adjust="local"):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    perf_dict = {'PI_curve':[],}
    verbose = False

    loaders, num_train, input_size, input_channel, n_class = get_loaders(args)
    input_dim = (input_channel, input_size, input_size)
    if len(loaders) == 3:
        train_loader, val_loader, test_loader = loaders
    else:
        train_loader, test_loader = loaders
        val_loader = None

    net = get_network(args.net, args.dataset, device, init=args.init)
    net = Sequential.from_concrete_network(net, input_dim, disconnect=True)
    print(net)

    assert os.path.isdir(args.load_model), "Here load-model should be the directory containing model.ckpt and Every_Epoch_Model directory."
    if "Every_Epoch_Model" in os.listdir(args.load_model):
        best_epoch = load_perf_from_json(args.load_model, "monitor.json")["best_ckpt_epoch"]
        n_epoch = load_perf_from_json(args.load_model, "train_args.json")["n_epochs"]
        for i in range(n_epoch):
            model_name = f"epoch_{i}.ckpt"
            net.load_state_dict(torch.load(os.path.join(args.load_model, "Every_Epoch_Model", model_name)))
            PI = PI_loop(net, args.test_eps, test_loader, device, n_class, args, relu_adjust)
            perf_dict["PI_curve"].append(PI)
            write_perf_to_json(perf_dict, args.load_model, "PI.json")

        perf_dict[f"final_PI"] = perf_dict[f"PI_curve"][best_epoch]

    else:
        model_name = "model.ckpt"
        net.load_state_dict(torch.load(os.path.join(args.load_model, model_name)))
        PI = PI_loop(net, args.test_eps, test_loader, device, n_class, args, relu_adjust)
        perf_dict[f"final_{relu_adjust}_PI"] = round(PI, 4)

    perf_dict["time"] = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    write_perf_to_json(perf_dict, args.load_model, "PI.json")
    
def run_BoxSize(args, normalize:bool=True):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    perf_dict = {}
    verbose = False

    loaders, num_train, input_size, input_channel, n_class = get_loaders(args)
    input_dim = (input_channel, input_size, input_size)
    if len(loaders) == 3:
        train_loader, val_loader, test_loader = loaders
    else:
        train_loader, test_loader = loaders
        val_loader = None

    net = get_network(args.net, args.dataset, device, init=args.init)
    net = Sequential.from_concrete_network(net, input_dim, disconnect=True)
    print(net)

    assert os.path.isdir(args.load_model), "Here load-model should be the directory containing model.ckpt and Every_Epoch_Model directory."

    model_name = "model.ckpt"
    net.load_state_dict(torch.load(os.path.join(args.load_model, model_name)))
    # vanilla, local, shirnk = PI_loop(net, args.test_eps, test_loader, device, n_class, args)
    margin = Margin_Loop(net, test_loader, device, n_class, args)
    bs = BoxSize_Loop(net, args.test_eps, test_loader, device, n_class, args)
    perf_dict[f"final_Boxsize"] = round(bs, 4)
    perf_dict[f"final_margin"] = round(margin, 4)
    perf_dict[f"Normalized_BS"] = round(bs / margin, 4)

    perf_dict["time"] = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    write_perf_to_json(perf_dict, args.load_model, "Boxsize.json")
    
def run_relu(args, eps:float=0):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    perf_dict = {}
    verbose = False

    loaders, num_train, input_size, input_channel, n_class = get_loaders(args)
    input_dim = (input_channel, input_size, input_size)
    if len(loaders) == 3:
        train_loader, val_loader, test_loader = loaders
    else:
        train_loader, test_loader = loaders
        val_loader = None

    net = get_network(args.net, args.dataset, device, init=args.init)
    net = Sequential.from_concrete_network(net, input_dim, disconnect=True)
    print(net)

    assert os.path.isdir(args.load_model), "Here load-model should be the directory containing model.ckpt and Every_Epoch_Model directory."

    model_name = "model.ckpt"
    net.load_state_dict(torch.load(os.path.join(args.load_model, model_name)))
    dead, unstable, active = relu_loop(net, eps, test_loader, device, args)
    perf_dict[f"dead_relu"] = round(dead, 4)
    perf_dict[f"unstable_relu"] = round(unstable, 4)
    perf_dict[f"active_relu"] = round(active, 4)

    perf_dict["time"] = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    write_perf_to_json(perf_dict, args.load_model, "relu.json")
    
def run_accu(args, dataset:str="train"):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    perf_dict = {}
    verbose = False

    loaders, num_train, input_size, input_channel, n_class = get_loaders(args)
    input_dim = (input_channel, input_size, input_size)
    if len(loaders) == 3:
        train_loader, val_loader, test_loader = loaders
    else:
        train_loader, test_loader = loaders
        val_loader = None

    net = get_network(args.net, args.dataset, device, init=args.init)
    net = Sequential.from_concrete_network(net, input_dim, disconnect=True)
    print(net)

    assert os.path.isdir(args.load_model), "Here load-model should be the directory containing model.ckpt and Every_Epoch_Model directory."

    model_name = "model.ckpt"
    net.load_state_dict(torch.load(os.path.join(args.load_model, model_name)))
    model_wrapper = BoxModelWrapper(net, nn.CrossEntropyLoss(), input_dim, device, args, None, min_eps_pgd=args.min_eps_pgd)

    if dataset == "train":
        dataloader = train_loader
    elif dataset == "test":
        dataloader = test_loader

    nat_accu, cert_accu, loss = test_loop(model_wrapper, args.test_eps, dataloader, device, args)
    perf_dict[f"{dataset}_nat_accu"] = round(nat_accu, 4)
    perf_dict[f"{dataset}_cert_accu"] = round(cert_accu, 4)
    perf_dict[f"{dataset}_loss"] = round(loss, 4)

    perf_dict["time"] = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    write_perf_to_json(perf_dict, args.load_model, f"{dataset}_IBP_{args.test_eps}.json")

def run_param_sign(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    perf_dict = {}

    loaders, num_train, input_size, input_channel, n_class = get_loaders(args)
    input_dim = (input_channel, input_size, input_size)

    net = get_network(args.net, args.dataset, device, init=args.init)
    net = Sequential.from_concrete_network(net, input_dim, disconnect=True)

    assert os.path.isdir(args.load_model), "Here load-model should be the directory containing model.ckpt and Every_Epoch_Model directory."

    model_name = "model.ckpt"
    net.load_state_dict(torch.load(os.path.join(args.load_model, model_name)))
    net = fuse_BN_wrt_Flatten(net, device, remove_all=True)
    print(net)

    for name, param in net.named_parameters():
        perf_dict[name] = (round((param > 0).sum().item() / param.numel(), 4), round((param < 0).sum().item() / param.numel(), 4), round((param == 0).sum().item() / param.numel(), 4))
    write_perf_to_json(perf_dict, args.load_model, "param_sign.json")

def main():
    args = get_args()
    seed_everything(args.random_seed)
    # run_PI(args)
    run_accu(args, dataset="test")
    # run_BoxSize(args)
    # run_relu(args)
    # run_param_sign(args)

if __name__ == '__main__':
    main()
