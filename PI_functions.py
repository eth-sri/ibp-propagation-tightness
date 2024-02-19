'''
Computing propagation tightness = (|mul(Wi)|@eps) / (mul(|Wi|)@eps) for a given network and input eps, where @ is matrix product.

This file contains two different but equivalent implementations of the function `compute_tightness`, the main difference being the way of computing |mul(Wi)|@eps:
    1. The original implementation, which first computes the sign of input dimensions in order to achieve maximum (i.e., the absolute value) and then use normal forward inference over these signed inputs to get |mul(Wi)|@eps = |mul(Wi) @ (SIGN*eps)| for proper vector SIGN, where * is element-wise product.
    2. A more efficient implementation, which uses backward propagation to compute |mul(Wi)|. Then it computes |mul(Wi)|@eps directly.

Both implementations are correct and have been tested against each other. The second implementation is able to get the intermediate result |mul(Wi)|. Both implementations have overall complexity O(KT), where K is the number of classes (or output dimensions) and T is the cost of a single normal network inference. Note that a vanilla forward computation of |mul(Wi)| will leads to O(DT) complexity, where D >> K is the input dimension, e.g., D=784 and K=10 for MNIST.

While the second implementation is arguably better, we keep the first implementation as all experiments in this paper is done via the first implementation which is developed originally. The second implementation is provided for future reference and potential improvement.

compute_tightness function has a key parameter `relu_adjust` which is used to adjust the ReLU activation pattern for the input. The following options are available:
    1. None: no adjustment, i.e., consider the ReLU layer as an identity function.
    2. "local": use the activation pattern at the original input as the adjustment, i.e., approximate the ReLU net with a DLN at the original input.
While other options, e.g., "shrink", are also implemented, they are not used and defined in the paper.
'''

import torch
import torch.nn.functional as F
from AIDomains.abstract_layers import Sequential, Flatten, Linear, ReLU, Conv2d, _BatchNorm, BatchNorm2d, BatchNorm1d, Normalization
from AIDomains.zonotope import HybridZonotope
from AIDomains.ai_util import construct_C
from AIDomains.wrapper import propagate_abs


from PARC_networks import get_network
from loaders import get_loaders
import argparse
from utils import seed_everything



def propagate_eps(input, net, C, abs:bool, relu_adjust=None):
    for i, layer in enumerate(net):
        if isinstance(layer, Normalization):
            input = input / layer.sigma # the weight matrix is all positive
        elif isinstance(layer, _BatchNorm):
            w = (layer.weight / torch.sqrt(layer.current_var + layer.eps)).view(layer.view_dim)
            if abs:
                input = input * w.abs()
            else:
                input = input * w
        elif isinstance(layer, Linear):
            if i != len(net) - 1:
                if abs:
                    input = F.linear(input, layer.weight.abs())
                else:
                    input = F.linear(input, layer.weight)
            else:
                # last linear, apply elision
                if abs:
                    elided_weight = torch.matmul(C, layer.weight).abs()
                else:
                    elided_weight = torch.matmul(C, layer.weight)
                input = torch.matmul(elided_weight, input.unsqueeze(-1)).squeeze(-1)
        elif isinstance(layer, Conv2d):
            if abs:
                input = F.conv2d(input, layer.weight.abs(), stride=layer.stride, padding=layer.padding, dilation=layer.dilation)
            else:
                input = F.conv2d(input, layer.weight, stride=layer.stride, padding=layer.padding, dilation=layer.dilation)
        elif isinstance(layer, Flatten):
            input = input.flatten(1, -1)
        elif isinstance(layer, ReLU):
            if relu_adjust is None:
                pass
            elif relu_adjust in ["local", "center"]:
                lb, ub = layer.bounds
                deactivation = ub < 0
                input[deactivation] = 0
            elif relu_adjust == "shrink":
                lb, ub = layer.bounds
                pre_size = ub - lb
                not_dead = ub > 0
                lb, ub = lb.clamp(min=0), ub.clamp(min=0)
                post_size = ub - lb
                input = input * (post_size.clamp(min=1e-8) / pre_size.clamp(min=1e-8)) * not_dead
            else:
                raise NotImplementedError(f"Unknown ReLU adjustment: {relu_adjust}")
        else:
            raise NotImplementedError(f"Unknown layer: {layer}")
    return input

def compute_tightness(net, batch_x, batch_y, eps, data_range=(0,1), num_classes:int=10, relu_adjust=None, detach_denom:bool=False, detach_num:bool=False, error_check:bool=False):
    '''
    Warning: this would destroy the previous grad and stored box bounds for the net
    '''
    input_eps = ((batch_x+eps).clamp(max=data_range[1]) - (batch_x-eps).clamp(min=data_range[0])) / 2
    num = input_eps.clone().detach()

    if batch_y is None:
        C = torch.eye(num_classes, device=batch_x.device).repeat(batch_x.shape[0], 1, 1)
    else:
        C = construct_C(num_classes, batch_y)

    # BN_layers = [layer for layer in net if isinstance(layer, _BatchNorm)]
    # original_stat = [layer.update_stat for layer in BN_layers]
    # for layer in BN_layers:
    #     layer.update_stat = False

    # set relu adjustment here
    # test status: correct. relu stat does not change inside this function after setting it below.
    net.reset_bounds()
    if relu_adjust == "local":
        # use the activation pattern at the original input as the adjustment
        with torch.no_grad():
            abs_input = HybridZonotope.construct_from_noise(batch_x, 0, domain="box")
            _ = net(abs_input)
    elif relu_adjust == "center":
        with torch.no_grad():
            center = ((batch_x+eps).clamp(max=data_range[1]) + (batch_x-eps).clamp(min=data_range[0])) / 2
            abs_input = HybridZonotope.construct_from_noise(center, 0, domain="box")
            _ = net(abs_input)
    elif relu_adjust == "shrink":
        # for unstable neurons, shrink the coefficient to ensure the same box size
        abs_input = HybridZonotope.construct_from_noise(batch_x, eps, domain="box")
        _ = net(abs_input)
    elif relu_adjust is None:
        pass
    else:
        raise NotImplementedError(f"Unknown ReLU adjustment: {relu_adjust}")

    # infer signs of numerator here
    with torch.enable_grad():
        num.requires_grad = True
        out = propagate_eps(num, net, C, abs=False, relu_adjust=relu_adjust)
        net.zero_grad()
        signs = []
        out_dim = out.shape[-1]
        for i in range(out_dim):
            num.grad = None
            # sum over batch because we only want the grad w.r.t. the batch eps which are unconnected
            # thus, the grad of the sum is their individual grad
            # test status: correct; tested via comparing the individual backward with it
            out[..., i].sum().backward(retain_graph=True) 
            signs.append(num.grad.sign())

    # compute the numerator
    # test status: no error found; tested via checking whether all num are the largest and positive
    num = []
    for i, sign in enumerate(signs):
        num_one_dim = propagate_eps(input_eps * sign, net, C, abs=False, relu_adjust=relu_adjust)
        num.append(num_one_dim)
    num = torch.diagonal(torch.stack(num, dim=-1), dim1=-2, dim2=-1)

    # compute the denominator
    # test status: correct; tested via comparing direct propagation on a Deep Linear Network
    # Numerical Problem with BN: result has <0.001% inconsistency
    denom = propagate_eps(input_eps, net, C, abs=True, relu_adjust=relu_adjust)

    if detach_num:
        num = num.detach()
    if detach_denom:
        denom = denom.detach()

    # print("num:", num)
    # print("denom:", denom)

    # abs_input = HybridZonotope.construct_from_noise(batch_x, eps, domain="box")
    # abs_out = net(abs_input, C=C)
    # lb, ub = abs_out.concretize()
    # print("real:", (ub-lb)/2)


    # for layer, stat in zip(BN_layers, original_stat):
    #     layer.update_stat = stat

    net.reset_bounds()
    
    ratio = num.clamp(min=1e-8) / denom.clamp(min=1e-8)

    if error_check and not (ratio <= 1.01).all():
        # numerical errors could lead to this;
        # enable error_check=True if this is strict
        mask = ratio > 1
        print(num[mask])
        print(denom[mask])
        torch.save(net, "buggie.ckpt")
        raise RuntimeError("PI > 1 detected.")
    return ratio

# ----- We provide a more efficient version of the above function by multiplication from back to front; correctness of the function below is tested via comparing the result to the implementation above. -----

# def backward_weight_calc(C, net, abs:bool, relu_adjust=None):
#     cur_W = C.clone().detach()
#     for rev_idx, layer in enumerate(net[::-1]):
#         if isinstance(layer, Linear):
#             if rev_idx == 0:
#                 cur_W = torch.matmul(cur_W, layer.weight)
#                 if abs:
#                     cur_W = cur_W.abs()
#             else:
#                 if abs:
#                     cur_W = torch.matmul(cur_W, layer.weight.abs())
#                 else:
#                     cur_W = torch.matmul(cur_W, layer.weight)
#         elif isinstance(layer, ReLU):
#             if relu_adjust in ["local", "center", "dead", "random_value_avg"]:
#                 lb, ub = layer.bounds
#                 activated = (lb >= 0).unsqueeze(1)
#                 cur_W = cur_W * activated.float()
#         elif isinstance(layer, Flatten):
#             in_dim = net[len(net)-rev_idx-2].output_dim
#             cur_W = cur_W.view(cur_W.shape[0], -1, *in_dim)
#         elif isinstance(layer, Conv2d):
#             # merge the batch dim and output dim
#             bs = cur_W.shape[0]
#             cur_W = cur_W.view(-1, *cur_W.shape[2:])
#             in_dim = net[len(net)-rev_idx-2].output_dim
#             w_padding = (
#                 in_dim[1]
#                 + 2 * layer.padding[0]
#                 - 1
#                 - layer.dilation[0] * (layer.weight.shape[-2] - 1)
#             ) % layer.stride[0]
#             h_padding = (
#                 in_dim[2]
#                 + 2 * layer.padding[1]
#                 - 1
#                 - layer.dilation[1] * (layer.weight.shape[-1] - 1)
#             ) % layer.stride[1]
#             output_padding = (w_padding, h_padding)

#             weight = layer.weight if not abs else layer.weight.abs()
#             cur_W = F.conv_transpose2d(cur_W, weight, stride=layer.stride, padding=layer.padding, output_padding=output_padding, groups=layer.groups, dilation=layer.dilation)

#             # unmerge the batch dim and output dim: leads to wasted view operation but should be OK
#             cur_W = cur_W.view(bs, -1, *cur_W.shape[1:])
#         elif isinstance(layer, Normalization):
#             # merge the batch dim and output dim
#             bs = cur_W.shape[0]
#             cur_W = cur_W.view(-1, *cur_W.shape[2:])
#             cur_W = cur_W / layer.sigma
#             # unmerge the batch dim and output dim: leads to wasted view operation but should be OK
#             cur_W = cur_W.view(bs, -1, *cur_W.shape[1:])
#         elif isinstance(layer, _BatchNorm):
#             # merge the batch dim and output dim
#             bs = cur_W.shape[0]
#             cur_W = cur_W.view(-1, *cur_W.shape[2:])
#             w = (layer.weight / torch.sqrt(layer.current_var + layer.eps)).view(layer.view_dim)
#             if abs:
#                 w = w.abs()
#             cur_W = cur_W * w
#             # unmerge the batch dim and output dim: leads to wasted view operation but should be OK
#             cur_W = cur_W.view(bs, -1, *cur_W.shape[1:])
#     cur_W = cur_W.flatten(start_dim=2).abs()
#     return cur_W

# def compute_tightness(net, batch_x, batch_y, eps:float=None, only_W:bool=False, detach_opt:bool=False, data_range=(0,1), num_classes:int=10, relu_adjust=None, num_samples:int=5, error_check:bool=False, verbose:bool=False):
#     '''
#     Compute end to end propagation tightness of the given network considering elision of the last layer, i.e., Interval Bound Propagation.

#     Warning: this would destroy the stored box bounds for the net
#     '''
#     if batch_y is None:
#         C = torch.eye(num_classes, device=batch_x.device).repeat(batch_x.shape[0], 1, 1)
#     else:
#         C = construct_C(num_classes, batch_y)

#     # set relu adjustment here
#     # test status: correct. relu stat does not change inside this function after setting it below.
#     net.reset_bounds()
#     if relu_adjust == "local":
#         # use the activation pattern at the original input as the adjustment
#         with torch.no_grad():
#             abs_input = HybridZonotope.construct_from_noise(batch_x, 0, domain="box")
#             _ = net(abs_input)
#     elif relu_adjust == "center":
#         # use the activation pattern at the center of the input as the adjustment
#         with torch.no_grad():
#             center = ((batch_x+eps).clamp(max=data_range[1]) + (batch_x-eps).clamp(min=data_range[0])) / 2
#             abs_input = HybridZonotope.construct_from_noise(center, 0, domain="box")
#             _ = net(abs_input)
#     elif relu_adjust == "dead":
#         # only deactivate the dead neurons
#         with torch.no_grad():
#             abs_input = HybridZonotope.construct_from_noise(batch_x, eps, domain="box")
#             _ = net(abs_input)    
#     elif relu_adjust is None:
#         pass
#     else:
#         raise NotImplementedError(f"Unknown ReLU adjustment: {relu_adjust}")
    
#     # Compute num_W = |\Prod_i W_i| and denom_W = \Prod_i |W_i|
#     if detach_opt:
#         with torch.no_grad():
#             num_W = backward_weight_calc(C, net, abs=False, relu_adjust=relu_adjust)
#     else:
#         num_W = backward_weight_calc(C, net, abs=False, relu_adjust=relu_adjust)
#     denom_W = backward_weight_calc(C, net, abs=True, relu_adjust=relu_adjust)
    
#     if only_W:
#         return num_W, denom_W
#     else:
#         assert eps is not None, "eps must be provided if only_W=False"

#     input_eps = ((batch_x+eps).clamp(max=data_range[1]) - (batch_x-eps).clamp(min=data_range[0])) / 2
#     input_eps = input_eps.flatten(start_dim=1).unsqueeze(-1)
#     num = torch.matmul(num_W, input_eps).squeeze(-1)
#     denom = torch.matmul(denom_W, input_eps).squeeze(-1)
#     ratio = num.clamp(min=1e-8) / denom.clamp(min=1e-8)

#     if error_check and not (ratio <= 1.01).all():
#         # numerical errors could lead to this;
#         # enable error_check=True if this is strict
#         mask = ratio > 1
#         print(num[mask])
#         print(denom[mask])
#         torch.save(net, "buggie.ckpt")
#         raise RuntimeError("PI > 1 detected.")
    
#     if verbose:
#         return ratio, num, denom
#     else:
#         return ratio


if __name__ == "__main__":
    seed_everything(0)
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.dataset = "mnist"
    args.train_batch = 128
    args.test_batch = 128
    args.grad_accu_batch = None
    args.frac_valid = None
    args.net = "cnn_3layer"
    args.init = "default"

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loaders, num_train, input_size, input_channel, n_class = get_loaders(args)
    train_loader, test_loader = loaders
    input_dim = (input_channel, input_size, input_size)

    net = get_network(args.net, args.dataset, device, init=args.init)
    net = Sequential.from_concrete_network(net, input_dim, disconnect=True)
    net.load_state_dict(torch.load("test_models/mnist/eps0.1/box_trained/cnn_3layer/init_fast/alpha5.0/fast_reg/model.ckpt"))

    print(net)

    eps = 0.3
    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device)
        compute_tightness(net, x, y, eps, relu_adjust=None) # relu_adjust="local" for the local tightness defined