# What can linearized neural networks *actually* say about generalization?

This is the source code to reproduce the experiments of the NeurIPS 2021 paper "[What can linearized neural networks actually say about generalization?](https://arxiv.org/abs/2106.06770)" by Guillermo Ortiz-Jimenez, Seyed-Mohsen Moosavi-Dezfooli and Pascal Frossard.

## Dependencies

To run the code, please install all its dependencies by running:
``` sh
$ pip install -r requirements.txt
```
This assumes that you have access to a Linux machine with an NVIDIA GPU with `CUDA>=11.1`. Otherwise, please check the instructions to install JAX with your setup in the [corresponding repository](https://github.com/google/jax#installation).

In general, all scripts are parameterized using [`hydra`](https://hydra.cc/docs/configure_hydra/intro/) and their configuration files can be found in the `config/` folder.

## Experiments

The repository contains code to reproduce the following experiments:

- [Spectral decomposition of the empirical NTK at initialization](#compute_ntk)
- [Training of linear and non-linear models on binary eigenfunctions of the NTK at initialization](#train_ntk)
- [Estimation of NADs using the NTK](#compute_nads)
- [Training of linear and non-linear models on linearly separable datasets given by the NADs](#train_nads)
- [Comparison of the training dynamics of linear models with kernels extracted at initialization and after non-linear pretraining](#pretrained_ntk_comparison)
- [Training of linear and non-linear models on CIFAR2](#train_cifar)

### Spectral decomposition of NTK <a id="compute_ntk"></a>

To generate our new benchmark, consisting on the eigenfunctions of the NTK at initialization, please run the python script `compute_ntk.py` selecting a desired model (e.g., `mlp`, `lenet` or `resnet18`) and supporting dataset (e.g., `cifar10` or `mnist`). This can be done by running
``` sh
$ python compute_ntk.py model=lenet data.dataset=cifar10
```
This script will save the eigenvalues, eigenfunctions and weights of the model under `artifacts/eigenfunctions/{data.dataset}/{model}/`.

For other configuration options, please consult the configuration file `config/compute-ntk/config.yaml`.

#### Warning
Take into account that, for large models, this computation can take very long. For example, it took us two days to compute the full eigenvalue decomposition of the NTK of one randomly initialized ResNet18 using 4 NVIDIA V100 GPUs. The estimation of eigenvectors for the MLP or the LeNet, on the other hand, can be done in a matter of minutes, depending on the number of GPUs available and the selected `batch_size`


### Training on binary eigenfunctions <a id="train_ntk"></a>

Once you have estimated the eigenfunctions of the NTK, you should be able to train on any of them. To that end, select the desired `label_idx` (i.e. eigenfunction index), model and dataset, and run

``` sh
$ python train_ntk.py label_idx=100 model=lenet data.dataset=cifar10 linearize=False
```
You can choose to train with the original non-linear network, or its linear approximation by specifying your choice with the flag `linearize`. For the non-linear models, this script also computes the final alignment of the end NTK with the target function, which it stores under `artifacts/eigenfunctions/{data.dataset}/{model}/alignment_plots/`

To see the different supported training options, please consult the configuration file `config/train-ntk/config.yaml`.

### Estimation of NADs <a id="compute_nads"></a>

We also provide code to compute the NADs of a CNN architecture (e.g., `lenet` or `resnet18`) using the alignment with the NTK at initialization. To do so, please run

``` sh
$ python compute_nads.py model=lenet
```
This script will save the eigenvalues, NADs and weights of the model under `artifacts/nads/{model}/`.

For other configuration options, please consult the configuration file `config/compute-nads/config.yaml`.

### Training on linearly separable datasets <a id="train_nads"></a>

Once you have estimated the NADs of a network, you should be able to train on linearly separable datasets with a single NAD as discriminative feature. To that end, select the desired `label_idx` (i.e. NAD index) and model, and run

``` sh
$ python train_nads.py label_idx=100 model=lenet linearize=False
```
You can choose to train with the original non-linear network, or its linear approximation by specifying your choice with the flag `linearize`.

To see the different supported training options, please consult the configuration file `config/train-nads/config.yaml`.

### Comparison of training dynamics with pretrained NTK <a id="pretrained_ntk_comparison"></a>

We also provide code to compare the training dynamics of the linearize network at initialization, and after non-linear pretraining, to estimate a particular eigenfunction of the NTK at initialization. To do this, please run
``` sh
$ python pretrained_ntk_comparison.py label_idx=100 model=lenet data.dataset=cifar10
```
To see the different supported training options, please consult the configuration file `config/pretrained_ntk_comparison/config.yaml`.

### Training on CIFAR2 <a id="train_cifar"></a>

Finally, you can train a neural network and its linearize approximation on the binary version of CIFAR10, i.e., CIFAR2. To do this, please run
``` sh
$ python train_cifar.py model=lenet linearize=False
```
To see the different supported training options, please consult the configuration file `config/binary-cifar/config.yaml`.

## Reference
If you use this code, please cite the following paper:

``` bibtex
@InCollection{Ortiz-JimenezNeurIPS2021,
  title = {What can linearized neural networks actually say about generalization?},
  author = {{Ortiz-Jimenez}, Guillermo and {Moosavi-Dezfooli}, Seyed-Mohsen and Frossard, Pascal},
  booktitle = {Advances in Neural Information Processing Systems 35},
  month = Dec,
  year = {2021}
}
```
