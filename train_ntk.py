import os

import hydra
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from flax.serialization import from_state_dict
from omegaconf import DictConfig, OmegaConf

from models.jax import get_model
from neural_kernels.ntk import get_ntk_alignment_fn, linearize_model
from neural_kernels.utils import unflatten_kernel_eigvecs
from train import sgd_train
from utils.misc import binarize_eigenfunctions, make_variables, params_mse_dist


def ntk_comparison(model, train_data, test_data, eigvecs_init, end_variables, cfg):
    num_train = train_data.shape[0]

    eigvecs_init = unflatten_kernel_eigvecs(eigvecs_init, num_classes=1)
    data = jnp.concatenate([train_data, test_data], axis=0)

    alignment_fn = get_ntk_alignment_fn(model=model, variables=end_variables, batch_size=128)
    _, end_params = end_variables.pop("params")

    ntk_alignment_train = alignment_fn(train_data, eigvecs_init[:, :num_train], end_params)
    ntk_alignment_test = alignment_fn(test_data, eigvecs_init[:, num_train:], end_params)
    ntk_alignment = alignment_fn(data, eigvecs_init, end_params)

    save_path = f"{hydra.utils.get_original_cwd()}/artifacts/eigenfunctions/{cfg.data.dataset}/{cfg.model.model_name}/alignment_plots"
    os.makedirs(save_path, exist_ok=True)

    plt.figure()
    plt.semilogy(ntk_alignment_train)
    plt.xlabel("Init eig-index")
    plt.ylabel("NTK alignment (train)")
    plt.savefig(f"{save_path}/eig_{cfg.label_idx}_train.pdf")
    plt.close()

    plt.figure()
    plt.semilogy(ntk_alignment_test)
    plt.xlabel("Init eig-index")
    plt.ylabel("NTK alignment (test)")
    plt.savefig(f"{save_path}/eig_{cfg.label_idx}_test.pdf")
    plt.close()

    plt.figure()
    plt.semilogy(ntk_alignment)
    plt.xlabel("Init eig-index")
    plt.ylabel("NTK alignment (all)")
    plt.savefig(f"{save_path}/eig_{cfg.label_idx}_all.pdf")
    plt.close()

    print(f"Alignment plots can be found in {save_path}")


def load_dataset_and_model(cfg):

    save_path = f"{hydra.utils.get_original_cwd()}/artifacts/eigenfunctions/{cfg.data.dataset}/{cfg.model.model_name}"

    data = jnp.load(f"{save_path}/data.npy", allow_pickle=True)
    eigvecs = jnp.load(f"{save_path}/eigvecs.npy", allow_pickle=True)

    train_ds = {
        "data": data[: cfg.num_train],
        "labels": binarize_eigenfunctions(eigvecs[cfg.label_idx, : cfg.num_train]),
    }
    test_ds = {
        "data": data[cfg.num_train :],
        "labels": binarize_eigenfunctions(eigvecs[cfg.label_idx, cfg.num_train :]),
    }

    model = get_model(**cfg.model)
    init_variables_state_dict = jnp.load(f"{save_path}/init_variables.npy", allow_pickle=True)[()]
    init_variables = model.init(jax.random.PRNGKey(0), jnp.zeros(cfg.shape))

    loaded_variables = from_state_dict(init_variables, init_variables_state_dict)

    return train_ds, test_ds, model, loaded_variables, eigvecs


@hydra.main(config_path="config/train_ntk", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    main_key = jax.random.PRNGKey(cfg.seed)
    train_key, state_key = jax.random.split(main_key, 2)

    train_ds, test_ds, model, init_variables, eigvecs = load_dataset_and_model(cfg)

    if cfg.linearize:
        apply_fn = linearize_model(model, init_variables)
    else:
        apply_fn = model.apply

    output_summary = sgd_train(
        train_ds=train_ds,
        test_ds=test_ds,
        apply_fn=apply_fn,
        init_variables=init_variables,
        key=train_key,
        **cfg.train,
    )

    end_state = output_summary["end_state"]
    init_state = output_summary["init_state"]
    end_variables = make_variables(end_state.target, end_state.model_state)

    print(f"Final test accuracy: {output_summary['test']['accuracy'] * 100} %%")

    dist_init = params_mse_dist(init_state.target, end_state.target)
    print(f"Distance to initialization: {dist_init}")

    if not cfg.linearize:
        print("Computing end alignments...")
        ntk_comparison(model, train_ds["data"], test_ds["data"], eigvecs, end_variables, cfg)


if __name__ == "__main__":
    main()
