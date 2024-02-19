# %%
import torch
import torch.nn as nn
import numpy as np
from AIDomains.abstract_layers import Sequential
from AIDomains.zonotope import HybridZonotope
import matplotlib.pyplot as plt
from typing import Iterable
from PI_functions import compute_tightness
from tqdm.auto import tqdm

from utils import seed_everything, write_perf_to_json
seed_everything(1)

# %%


def get_transform_matrix(d, k, orthogonal:bool=False):
    # construct transform matrix
    W = np.random.randn(d, k)
    if orthogonal:
        u, s, vh = np.linalg.svd(W, full_matrices=False)
        W = u @ vh
    return W

def construct_data(d, k, N, scaling:float=1):
    W = get_transform_matrix(d, k, orthogonal=True)
    embedding = np.random.randn(N, k)
    X = embedding @ W.T
    X = torch.from_numpy(X).float() * scaling
    W = torch.from_numpy(W).float()
    return X, W

def get_model(d, w, relu:bool=False):
    if relu:
        model = nn.Sequential(
            nn.Linear(d, w),
            nn.ReLU(),
            nn.Linear(w, d)
        )
    else:
        model = nn.Sequential(
            nn.Linear(d, w, bias=False),
            nn.Linear(w, d, bias=False)
        )
    model = Sequential.from_concrete_network(model, input_dim=(d, ), disconnect=True)
    return model

def load_model(W1, W2, model):
    model[0].weight.data = W1
    model[-1].weight.data = W2
    return model

def compute_reconstruction_loss_and_avg_out_radius(X, eps:float, model):
    abs_X = HybridZonotope.construct_from_bounds(X-eps, X+eps, domain="box")
    lb, ub = model(abs_X).concretize()
    diff = torch.maximum((lb-X).abs(), (ub-X).abs())
    IBP_loss = torch.sqrt((diff**2).sum(dim=-1)).mean()
    std_loss = torch.sqrt((((ub + lb) / 2 - X)**2).sum(dim=-1)).mean()
    out_radius = ((ub - lb) / (2*max(eps, 1e-8))).mean()
    return IBP_loss, std_loss, out_radius

def compute_L1_reconstruction_loss(X, eps, model):
    abs_X = HybridZonotope.construct_from_bounds(X-eps, X+eps, domain="box")
    lb, ub = model(abs_X).concretize()
    diff = torch.maximum((lb-X).abs(), (ub-X).abs())
    IBP_loss = diff.sum(dim=-1).mean()
    return IBP_loss

def compute_optimal_loss_and_radius(X, eps:float, model):
    '''
    Will ignore ReLU layer if model is not DLN
    '''
    W1 = model[0].weight.data
    W2 = model[-1].weight.data
    # W = W2 @ W1
    # X_hat = X @ W.T
    # opt_radius =  torch.ones_like(X, device=X.device) * eps @ W.T.abs()
    X_hat = X @ W1.T @ W2.T
    opt_radius =  torch.ones_like(X, device=X.device) * eps @ (W1.T @ W2.T).abs()
    lb, ub = X_hat - opt_radius, X_hat + opt_radius

    '''checked correct bounds'''
    # single_layer = Sequential.from_concrete_network(nn.Sequential(nn.Linear(d, d, bias=False)), input_dim=(d, )).to("cuda")
    # single_layer[0].weight.data = W
    # abs_X = HybridZonotope.construct_from_bounds(X-eps, X+eps, domain="box")
    # true_lb, true_ub = single_layer(abs_X).concretize()

    diff = torch.maximum((lb-X).abs(), (ub-X).abs())
    opt_loss = torch.sqrt((diff**2).sum(dim=-1)).mean()

    opt_radius = (opt_radius / max(eps, 1e-8)).mean()
    return opt_loss, opt_radius

def train(model, X, eps:float, device, anneal_epoch:int=40, num_epoch:int=100, lr_milestones:Iterable=[50, 150], lr=5e-3):
    '''
    IBP training; For standard training, use eps=0
    '''
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_curve, radius_curve = [], []
    for epoch_idx in range(num_epoch):
        current_eps = min(epoch_idx/max(anneal_epoch, 1e-3), 1) * eps
        loss, std_loss, out_radius = compute_reconstruction_loss_and_avg_out_radius(X, current_eps, model)
        loss_curve.append(loss.item())
        radius_curve.append(out_radius.item())
        loss.backward()
        optimizer.step()
        if epoch_idx in lr_milestones:
            for g in optimizer.param_groups:
                g["lr"] *= 0.1
    return model, loss_curve, radius_curve

def run(d:int, k:int, w:int, N:int, train_eps:float, test_eps:float, relu:bool, use_PCA_weight:bool, verbose:bool=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    X, W = construct_data(d, k, N)
    model = get_model(d, w, relu=relu)
    X, model, W = X.to(device), model.to(device), W.to(device)
    if use_PCA_weight:
        model = load_model(W.T, W, model)
    else:
        model, loss_curve, radius_curve = train(model, X, train_eps, device)
        if verbose:
            print(loss_curve)
    if relu:
        opt_loss, opt_radius = None, None
    else:
        opt_loss, opt_radius = compute_optimal_loss_and_radius(X, test_eps, model)
        opt_loss, opt_radius = round(opt_loss.item(), 4), round(opt_radius.item(), 4)
    with torch.no_grad():
        IBP_loss, std_loss, IBP_radius = compute_reconstruction_loss_and_avg_out_radius(X, test_eps, model)
        # TODO:
        IBP_loss = compute_L1_reconstruction_loss(X, test_eps, model)
        relu_adjust = "local" if relu else None
        tightness = compute_tightness(model, X, None, 1e-5, data_range=(-1e8, 1e8), num_classes=d, relu_adjust=relu_adjust).mean()
    return round(IBP_loss.item(), 4), round(std_loss.item(), 4), round(IBP_radius.item(), 4), round(tightness.item(), 4), opt_loss, opt_radius

eps = 0.01
N = 1000
k = 40
d = 100
k_equal_w = True
print(eps, N, k, d, k_equal_w)

w_list = list(range(30, 0, -1))
perf = {
    "width": w_list,
    "optimal": [],
    "IBP_DLN": [],
    "STD_DLN": [],
    "IBP_relu": [],
    "STD_relu": []
}

for w in tqdm(w_list):
    if k_equal_w:
        k = w
    # Optimal
    IBP_loss, std_loss, IBP_radius, tightness, opt_loss, opt_radius = run(d, k, w, N, None, eps, relu=False, use_PCA_weight=True)
    perf["optimal"].append((IBP_loss, std_loss, tightness, opt_loss, opt_radius, IBP_radius))
    # print(IBP_radius / w)

    # # IBP DLN
    # IBP_loss, std_loss, IBP_radius, tightness, opt_loss, opt_radius = run(d, k, w, N, eps, eps, relu=False, use_PCA_weight=False)
    # perf["IBP_DLN"].append((IBP_loss, std_loss, tightness, opt_loss, opt_radius, IBP_radius))
    # print(IBP_loss, IBP_radius)

    # # std DLN
    # IBP_loss, std_loss, IBP_radius, tightness, opt_loss, opt_radius = run(d, k, w, N, 0.0, eps, relu=False, use_PCA_weight=False)
    # perf["STD_DLN"].append((IBP_loss, std_loss, tightness, opt_loss, opt_radius, IBP_radius))

    # # IBP ReLU
    # IBP_loss, std_loss, IBP_radius, tightness, opt_loss, opt_radius = run(d, k, w, N, eps, eps, relu=True, use_PCA_weight=False)
    # perf["IBP_relu"].append((IBP_loss, std_loss, tightness, opt_loss, opt_radius, IBP_radius))

    # # std ReLU
    # IBP_loss, std_loss, IBP_radius, tightness, opt_loss, opt_radius = run(d, k, w, N, 0.0, eps, relu=True, use_PCA_weight=False)
    # perf["STD_relu"].append((IBP_loss, std_loss, tightness, opt_loss, opt_radius, IBP_radius))

write_perf_to_json(perf, "./", f"IBP_embedding{'_k=w' if k_equal_w else ''}.json")