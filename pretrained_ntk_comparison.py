import hydra
import jax
import jax.numpy as jnp
from flax.serialization import from_state_dict
from omegaconf import DictConfig, OmegaConf

from models.jax import get_model
from neural_kernels.ntk import linearize_diff_model
from train import sgd_train
from utils.misc import binarize_eigenfunctions, make_variables


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


@hydra.main(config_path="config/pretrained_ntk_comparison", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    main_key = jax.random.PRNGKey(cfg.seed)
    train_key, state_key = jax.random.split(main_key, 2)

    train_ds, test_ds, model, init_variables, eigvecs = load_dataset_and_model(cfg)

    # Train models
    print("Training neural network to rotate kernel...")
    nn_summary = sgd_train(
        train_ds=train_ds,
        test_ds=test_ds,
        apply_fn=model.apply,
        init_variables=init_variables,
        key=train_key,
        **cfg.train,
    )

    print("Linearizing model...")
    linear_model_fn = linearize_diff_model(model, init_variables)

    print("Training unrotated kernel...")
    ntk0_summary = sgd_train(
        train_ds=train_ds,
        test_ds=test_ds,
        apply_fn=linear_model_fn,
        init_variables=init_variables,
        key=train_key,
        **cfg.train,
    )

    print("Linearizing pretrained model...")

    init_model_state, _ = init_variables.pop(
        "params"
    )  # We reinitalize batch norm to not introduce an unfair advantage
    pretrained_variables = make_variables(nn_summary["end_state"].target, init_model_state)
    pretrained_linear_model_fn = linearize_diff_model(model, pretrained_variables)

    print("Training pretrained kernel...")
    ntk_pretrained_summary = sgd_train(
        train_ds=train_ds,
        test_ds=test_ds,
        apply_fn=pretrained_linear_model_fn,
        init_variables=pretrained_variables,
        key=train_key,
        **cfg.train,
    )

    print(f"(Non-linear) Final test accuracy: {nn_summary['test']['accuracy'] * 100} %%")
    print(f"(Init linear) Final test accuracy: {ntk0_summary['test']['accuracy'] * 100} %%")
    print(
        f"(Pretrained linear) Final test accuracy: {ntk_pretrained_summary['test']['accuracy'] * 100} %%"
    )


if __name__ == "__main__":
    main()
