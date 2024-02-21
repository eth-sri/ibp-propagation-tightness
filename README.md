# Propagation Tightness of IBP

This is the code repository for the ICLR'24 paper [Understanding Certified Training with Interval Bound Propagation](https://arxiv.org/abs/2306.10426).

## Environment

We use Python version 3.9 and PyTorch 1.12.1, which can be installed in a conda environment as follows:

```console
conda create --name ibp_tightness python=3.9
conda activate ibp_tightness
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
```

To install further requirements please run 
```console
pip install -r requirements.txt
```


## Core Files

- ```PI_functions.py```: implements the computation of propagation tightness. MAIN contribution.
- ```theory_local_approximation.py``` implements the experiments about the approximation error between local propagation tightness and true propagation tightness. Figure 2 and Figure 14 in the paper.
- ```relu_initialization.py```: implements the initialization experiment. Figure 3 in the paper.
- ```IBP_embedding_exp.py```: implements the embedding experiment. Figure 4 in the paper.
- ```get_stat.py```: implements various statistics computations for model checkpoints.
- ```args_factory.py```, ```attacks.py```, ```loaders.py```, ```mix_train.py```, ```mnbab_certify.py```, ```PARC_networks.py```, ```regularization.py```, ```torch_model_wrapper.py``` and ```util.py``` are adapted from the implementation of TAPS (https://arxiv.org/abs/2305.04574, https://github.com/eth-sri/taps). They are minimally revised to incorporate the computation of propagation tightness.


We comment on the top of the file ```PI_functions.py``` about the details of implementing propagation tightness computation.

## Commands for Experiment

To reproduce all our experiments please run the following scripts 

- ```train_cifar_arch.sh``` reproduces the depth and width experiment.
- ```train_cifar_eps.sh``` reproduces the experiment investigating the dependence of tightness on training perturbation magnitude epsilon.
- ```train_cifar_SABR``` reproduces the experiment investigating tightness for SABR-trained networks.

## Pretrained Models

As training all models takes some time, we release pre-trained models [here](https://mega.nz/file/GeRlmZyK#K-JGtFPdZ8HA3V3qHEimTHfOipk6fajna7yEbBSmD5Y). 
Due to space constraints on our hosting, we only include models for one random seed (still 10GB) for the width and depth experiments. We also include 4x model for MNIST eps=0.3 and 2x model for CIFAR eps=1/255 where enlarging width leads to SOTA performance.

License and Copyright
---------------------

* Copyright (c) 2024 [Secure, Reliable, and Intelligent Systems Lab (SRI), Department of Computer Science ETH Zurich](https://www.sri.inf.ethz.ch/)
* Licensed under the [Apache License](https://www.apache.org/licenses/LICENSE-2.0)