import torch
import torch.nn as nn
from args_factory import get_args
from loaders import get_loaders
from utils import Scheduler, Statistics
from PARC_networks import get_network
from utils import write_perf_to_json, load_perf_from_json, fuse_BN, seed_everything
from AIDomains.abstract_layers import Sequential
import os

def key_transform(COLT_key):
    ks = COLT_key.lstrip('blocks.layers.').split('.')
    idx, fun = ks[0], ks[-1]
    k = f'{idx}.{fun}'
    return k

def run(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loaders, num_train, input_size, input_channel, n_class = get_loaders(args)
    args.num_classes = n_class
    input_dim = (input_channel, input_size, input_size)

    net = get_network(args.net, args.dataset, device, init=args.init)
    print(net)

    param = torch.load(args.load_model)

    transformed_param = {}
    transformed_param['0.mean'] = net.state_dict()['0.mean']
    transformed_param['0.std'] = net.state_dict()['0.std']

    for k, v in param.items():
        k = key_transform(k)
        if k.split(".")[-1] == "deepz_lambda":
            continue
        transformed_param[k] = v
    net.load_state_dict(transformed_param)

    net = Sequential.from_concrete_network(net, input_dim, disconnect=True)
    print(net)

    os.makedirs(args.save_dir)
    torch.save(net.state_dict(), os.path.join(args.save_dir, "model.ckpt"))

def main():
    args = get_args()
    seed_everything(args.random_seed)
    run(args)



if __name__ == '__main__':
    main()
