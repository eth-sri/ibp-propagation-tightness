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
from regularization import compute_fast_reg, compute_vol_reg, compute_L1_reg, compute_neg_reg
import time
from datetime import datetime
from AIDomains.abstract_layers import Sequential, Flatten, Linear, ReLU, Conv2d

import warnings
warnings.filterwarnings("ignore")

from get_stat import PI_loop, relu_loop, test_loop




def train_loop(model_wrapper, eps_scheduler, pgd_scheduler, train_loader, epoch_idx, optimizer, device, args, verbose:bool=False):
    model_wrapper.net.train()

    if not args.use_vanilla_ibp:
        if not args.no_ibp_anneal:
            use_vanilla_ibp = True if epoch_idx < args.end_epoch_eps else False
        else:
            use_vanilla_ibp = False
    else:
        use_vanilla_ibp = True

    enforce_eval = False

    pgd_weight = 0 if pgd_scheduler is None else pgd_scheduler.getcurrent(epoch_idx * len(train_loader))
    model_wrapper.current_pgd_weight = pgd_weight
    fast_reg = args.fast_reg and epoch_idx < args.end_epoch_eps
    print("Fast reg:", fast_reg)
    fastreg_stat, nat_accu_stat, cert_accu_stat, loss_stat = Statistics.get_statistics(4)
    model_wrapper.store_box_bounds = True

    if args.verbose_first_epoch and epoch_idx == 0:
        # epoch_perf = {"cert_loss_curve":[], "fast_reg_curve":[], "vanilla_PI_curve":[], "local_PI_curve":[], "shrink_PI_curve":[], "center_PI_curve":[]}
        epoch_perf = {"cert_loss_curve":[], "fast_reg_curve":[], "local_PI_curve":[]}


    pbar = tqdm(train_loader)
    for batch_idx, (x, y) in enumerate(pbar):
        x, y = x.to(device), y.to(device)
        eps = eps_scheduler.getcurrent(epoch_idx * len(train_loader) + batch_idx)
        pgd_weight = 1 if pgd_scheduler is None else pgd_scheduler.getcurrent(epoch_idx * len(train_loader) + batch_idx)
        model_wrapper.current_pgd_weight = pgd_weight
        optimizer.zero_grad()
        (loss, nat_loss, cert_loss), (nat_accu, cert_accu), (is_nat_accu, is_cert_accu) = model_wrapper.common_step(x, y, eps, args.train_steps, use_vanilla_ibp, enforce_eval=enforce_eval, summary_accu_stat=False)
        if verbose:
            print(nat_accu, cert_accu, loss.item())
        

        loss_stat.update(loss.item(), len(x))

        if fast_reg:
            # add fast reg to the loss
            reg_eps = max(eps, args.min_eps_reg)
            if reg_eps != eps:
                # recompute bounds for fast reg
                model_wrapper.net.reset_bounds()
                model_wrapper.get_cert_performance((x-eps).clamp(min=0), (x+eps).clamp(max=1), y, reg_eps, args.train_steps, use_vanilla_ibp=True)
            reg = args.reg_lambda * (1 - reg_eps/eps_scheduler.end_value) * compute_fast_reg(model_wrapper.net, reg_eps)
            loss = loss + reg
            fastreg_stat.update(reg.item(), len(x))
        if args.L1_reg > 0:
            loss = loss + args.L1_reg * compute_L1_reg(model_wrapper.net)
        neg_reg = 0
        if args.neg_reg > 0:
            neg_reg = args.neg_reg * compute_neg_reg(model_wrapper.net)
            loss = loss + neg_reg

        vol_reg = 0
        if args.IBPR_reg > 0:
            vol_reg = args.IBPR_reg * compute_vol_reg(model_wrapper.net, x, eps, recompute_box=True, min_reg_eps=args.min_eps_reg, max_reg_eps=args.test_eps) / len(x) * model_wrapper.current_pgd_weight
            loss = loss + vol_reg

        if args.verbose_first_epoch and epoch_idx == 0 and ((batch_idx % int(args.verbose_gap * len(train_loader))) == 0):
            epoch_perf["cert_loss_curve"].append(loss_stat.last)
            epoch_perf["fast_reg_curve"].append(fastreg_stat.last)
            # will set net.eval but not destroy BN
            # vanilla, local, shrink, center = PI_loop(model_wrapper.net, max(eps, 1e-8), train_loader, device, args.num_classes, args)
            local = PI_loop(model_wrapper.net, max(eps, 1e-8), train_loader, device, args.num_classes, args)

            model_wrapper.net.train()
            # epoch_perf["vanilla_PI_curve"].append(vanilla)
            epoch_perf["local_PI_curve"].append(local)
            # epoch_perf["shrink_PI_curve"].append(shrink)
            # epoch_perf["center_PI_curve"].append(center)

            write_perf_to_json(epoch_perf, args.save_root, "first_epoch.json")


        model_wrapper.net.reset_bounds()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_wrapper.net.parameters(), args.grad_clip)
        optimizer.step()


        nat_accu_stat.update(nat_accu, len(x))
        cert_accu_stat.update(cert_accu, len(x))
        postfix_str = f"nat_accu: {nat_accu_stat.avg:.3f}, cert_accu: {cert_accu_stat.avg:.3f}"
        if vol_reg > 0:
            postfix_str += f", vol/total: {vol_reg/loss:.3E}"
        pbar.set_postfix_str(postfix_str)
    print("Average FastReg loss:", fastreg_stat.avg)

    return nat_accu_stat.avg, cert_accu_stat.avg, eps, fastreg_stat.avg, loss_stat.avg



def run(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    perf_dict = {'val_nat_curve':[], 'val_cert_curve':[], 'val_loss_curve':[], 'train_nat_curve':[], 'train_cert_curve':[], 'train_loss_curve':[], 'lr_curve':[], "eps_curve":[], 'fastreg_avg':[]}
    # PI_dict = {"vanilla_PI_curve": [], "local_PI_curve":[], "shrink_PI_curve":[], "center_PI_curve":[]}
    PI_dict = {"local_PI_curve":[]}
    relu_dict = {"dead_relu_curve":[], "active_relu_curve":[], "unstable_relu_curve":[]}

    perf_dict = perf_dict | PI_dict | relu_dict
    perf_dict["start_time"] = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    verbose = False

    loaders, num_train, input_size, input_channel, n_class = get_loaders(args)
    args.num_classes = n_class
    input_dim = (input_channel, input_size, input_size)
    if len(loaders) == 3:
        train_loader, val_loader, test_loader = loaders
        perf_dict['model_selection'] = True
    else:
        train_loader, test_loader = loaders
        val_loader = None
        perf_dict['model_selection'] = True

    pgd_scheduler = None
    if args.use_adv_training:
        mode = "adv_trained"
        pgd_scheduler = Scheduler(args.start_epoch_pgd_weight*len(train_loader), args.end_epoch_pgd_weight*len(train_loader), args.pgd_weight_start, args.pgd_weight_end, mode="linear", s=len(train_loader))
    elif args.use_vanilla_ibp:
        mode = f"{'small_' if args.use_small_box else ''}box_trained"
    else:
        mode = f"{'small_' if args.use_small_box else ''}TAPS_trained"
        pgd_scheduler = Scheduler(args.start_epoch_pgd_weight*len(train_loader), args.end_epoch_pgd_weight*len(train_loader), args.pgd_weight_start, args.pgd_weight_end, mode="linear", s=len(train_loader))

    if args.no_anneal:
        # use const eps == train_eps
        eps_scheduler = Scheduler(args.start_epoch_eps*len(train_loader), args.end_epoch_eps*len(train_loader), args.train_eps, args.train_eps, "linear", s=len(train_loader))
    else:
        # use smooth eps annealing
        # TODO: recover smooth
        if args.schedule == "smooth":
            eps_scheduler = Scheduler(args.start_epoch_eps*len(train_loader), args.end_epoch_eps*len(train_loader), args.eps_start, args.eps_end, "smooth", s=len(train_loader))
        elif args.schedule == "linear":
            eps_scheduler = Scheduler(args.start_epoch_eps*len(train_loader), args.end_epoch_eps*len(train_loader), args.eps_start, args.eps_end, "linear")
        elif args.schedule == "step":
            eps_scheduler = Scheduler(args.start_epoch_eps*len(train_loader), args.end_epoch_eps*len(train_loader), args.eps_start, args.eps_end, "step", s=args.step_epoch*len(train_loader))
        elif args.schedule == "step_linear":
            eps_scheduler = Scheduler(args.start_epoch_eps*len(train_loader), args.end_epoch_eps*len(train_loader), args.eps_start, args.eps_end, "step_linear", s=20*len(train_loader), c=0.5)
        else:
            raise NotImplementedError(f"Unknown schedule: {args.schedule}")


    if args.fast_reg and not args.use_adv_training:
        # net = get_network(args.net, args.dataset, device, init=True, init_until=args.block_sizes[0])
        net = get_network(args.net, args.dataset, device, init=args.init)
    else:
        net = get_network(args.net, args.dataset, device, init=args.init)
    # summary(net, (input_channel, input_size, input_size))
    net = Sequential.from_concrete_network(net, input_dim, disconnect=True)

    if args.load_model:
        net.load_state_dict(torch.load(args.load_model))
        print("Loaded:", args.load_model)

    if args.use_vanilla_ibp:
        block_sizes = None
        # block_sizes = args.block_sizes
        # net = fuse_BN_wrt_Flatten(net, device, num_layer_after_flatten=args.layers_after_flatten_to_fuse)
    else:
        # net = add_BN_wrt_Flatten(net, device, num_layer_after_flatten=args.layers_after_flatten_to_fuse)
        # net = fuse_BN_wrt_Flatten(net, device, num_layer_after_flatten=args.layers_after_flatten_to_fuse)
        block_sizes = args.block_sizes


    print(net)

    eps_str = f"eps{args.test_eps:.5g}"
    alpha_str = f"alpha{args.alpha_box}" if not args.no_ibp_reg else "no_ibp_reg"
    save_root = os.path.join(args.save_dir, args.dataset, eps_str, mode, args.net, f"init_{args.init}", alpha_str, f"certW_{args.cert_weight}")
    save_root = os.path.join(save_root, f"anneal_{args.end_epoch_eps - args.start_epoch_eps}")
    if args.fast_reg:
        save_root = os.path.join(save_root, "fast_reg")
    if args.no_ibp_anneal:
        save_root = os.path.join(save_root, "no_ibp_anneal")
    if args.use_small_box:
        save_root = os.path.join(save_root, f"lambda_{args.eps_shrinkage}")
    if args.load_model:
        save_root = os.path.join(save_root, "from_pretrain")

    if not (args.use_vanilla_ibp or args.use_adv_training):
        if args.use_single_estimator:
            save_root = os.path.join(save_root, "single_estimator")
        save_root = os.path.join(save_root, f"last_block_{args.block_sizes[-1]}")

    if args.neg_reg > 0:
        save_root = os.path.join(save_root, f"neg_{args.neg_reg}")
    if args.IBPR_reg > 0:
        save_root = os.path.join(save_root, f"IBPR_{args.IBPR_reg}")
    if args.L1_reg > 0:
        save_root = os.path.join(save_root, f"L1_{args.L1_reg}")

    args.save_root = save_root
    os.makedirs(save_root, exist_ok=True)
    print("The model will be saved at:", save_root)

    if args.use_adv_training:
        model_wrapper = PGDModelWrapper(net, nn.CrossEntropyLoss(), input_dim, device, args)
    elif args.use_small_box:
        if args.grad_accu_batch is None:
            model_wrapper = SmallBoxModelWrapper(net, nn.CrossEntropyLoss(), input_dim, device, args, block_sizes, min_eps_pgd=args.min_eps_pgd, eps_shrinkage=args.eps_shrinkage)
        else:
            model_wrapper = GradAccuSmallBoxModelWrapper(net, nn.CrossEntropyLoss(), input_dim, device, args, block_sizes, min_eps_pgd=args.min_eps_pgd, eps_shrinkage=args.eps_shrinkage)
    else:
        if args.grad_accu_batch is None:
            model_wrapper = BoxModelWrapper(net, nn.CrossEntropyLoss(), input_dim, device, args, block_sizes, min_eps_pgd=args.min_eps_pgd)
        else:
            model_wrapper = GradAccuBoxModelWrapper(net, nn.CrossEntropyLoss(), input_dim, device, args, block_sizes, min_eps_pgd=args.min_eps_pgd)

    param_list = set(model_wrapper.net.parameters()) - set(model_wrapper.net[0].parameters()) # exclude normalization

    # The main training loop
    lr = args.lr
    best_cert_accu = -1
    if args.opt == 'adam':
        optimizer = torch.optim.Adam(param_list, lr=lr)
    elif args.opt == 'sgd':
        optimizer = torch.optim.SGD(param_list, lr=lr)
    else:
        raise ValueError(f"{args.opt} not supported.")

    lr_schedular = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_milestones, gamma=0.2)
    model_wrapper.detach_ibp = False
    model_wrapper.current_pgd_weight = args.pgd_weight_start


    train_start_time = time.time()
    for epoch_idx in range(args.n_epochs):
        print("Epoch", epoch_idx)
        train_nat_accu, train_cert_accu, eps, fast_reg_avg, train_loss = train_loop(model_wrapper, eps_scheduler, pgd_scheduler, train_loader, epoch_idx, optimizer, device, args, verbose=verbose)
        perf_dict["fastreg_avg"].append(fast_reg_avg)

        # if eps == args.test_eps:
        print(f"train_nat_accu: {train_nat_accu: .4f}, train_cert_accu: {train_cert_accu: .4f}, train_loss:{train_loss: .4f}")
        perf_dict['train_nat_curve'].append(train_nat_accu)
        perf_dict['train_cert_curve'].append(train_cert_accu)
        perf_dict["train_loss_curve"].append(train_loss)

        lr_schedular.step()
        lr = lr_schedular.get_last_lr()[0]

        eps = min(eps, args.test_eps)
        print("current eps:", eps)
        print("current pgd_weight:", model_wrapper.current_pgd_weight)

        val_nat_accu, val_cert_accu, val_loss = test_loop(model_wrapper, eps, val_loader if val_loader is not None else test_loader, device, args)
        print(f"val_nat_accu: {val_nat_accu: .4f}, val_cert_accu: {val_cert_accu: .4f}, val_loss:{val_loss: .4f}")
        perf_dict['val_nat_curve'].append(val_nat_accu)
        perf_dict['val_cert_curve'].append(val_cert_accu)
        perf_dict["val_loss_curve"].append(val_loss)
        perf_dict['lr_curve'].append(lr)
        perf_dict["eps_curve"].append(eps)

        #### ---- PI loop below ----
        # vanilla, local, shrink, center = PI_loop(model_wrapper.net, max(eps, 1e-8), val_loader if val_loader is not None else test_loader, device, args.num_classes, args)
        local = PI_loop(model_wrapper.net, max(eps, 1e-5), val_loader if val_loader is not None else test_loader, device, args.num_classes, args)

        # print(f"Vanilla: {vanilla}; Local: {local}; Shrink: {shrink}; Center: {center}")
        print(f"Local PI: {local}")

        # perf_dict["vanilla_PI_curve"].append(vanilla)
        perf_dict["local_PI_curve"].append(local)
        # perf_dict["shrink_PI_curve"].append(shrink)
        # perf_dict["center_PI_curve"].append(center)

        #### ---- relu loop below ----
        dead, unstable, active = relu_loop(model_wrapper.net, max(eps, 1e-8), val_loader if val_loader is not None else test_loader, device, args)
        perf_dict["dead_relu_curve"].append(dead)
        perf_dict["unstable_relu_curve"].append(unstable)
        perf_dict["active_relu_curve"].append(active)
        print(f"Dead: {dead:.3f}; Unstable: {unstable:.3f}; Active: {active:.3f}")

        if eps == args.test_eps:
            if val_cert_accu > best_cert_accu:
                if perf_dict["model_selection"]:
                    # Select model based on val cert accu only when a val loader is present
                    torch.save(model_wrapper.net.state_dict(), os.path.join(save_root, "model.ckpt"))
                    print("New checkpoint saved.")
                best_cert_accu = val_cert_accu
                perf_dict["best_ckpt_epoch"] = epoch_idx
                perf_dict["best_val_cert_accu"] = best_cert_accu

        write_perf_to_json(perf_dict, save_root, "monitor.json")
        write_perf_to_json(args.__dict__, save_root, "train_args.json")

        if args.save_every_epoch:
            os.makedirs(os.path.join(save_root, "Every_Epoch_Model"), exist_ok=True)
            torch.save(model_wrapper.net.state_dict(), os.path.join(save_root, "Every_Epoch_Model", f"epoch_{epoch_idx}.ckpt"))

    if not perf_dict["model_selection"]:
        # No model selection. Save the final model.
        torch.save(model_wrapper.net.state_dict(), os.path.join(save_root, "model.ckpt"))

    # test for the best ckpt
    print("Best val cert_accu:", best_cert_accu)
    print("-"*10 + f"Model Selection: {perf_dict['model_selection']}. Testing selected checkpoint." + "-"*10)
    model_wrapper.net.load_state_dict(torch.load(os.path.join(save_root, "model.ckpt")))
    test_nat_accu, test_cert_accu, loss = test_loop(model_wrapper, args.test_eps, test_loader, device, args)
    print(f"test_nat_accu: {test_nat_accu: .4f}, test_cert_accu: {test_cert_accu: .4f}")
    perf_dict["test_nat_accu"] = test_nat_accu
    perf_dict["test_cert_accu"] = test_cert_accu
    perf_dict["total_time"] = time.time() - train_start_time
    perf_dict["end_time"] = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    
    write_perf_to_json(perf_dict, save_root, "monitor.json")
    write_perf_to_json(args.__dict__, save_root, "train_args.json")


def main():
    args = get_args()
    seed_everything(args.random_seed)
    run(args)



if __name__ == '__main__':
    main()
