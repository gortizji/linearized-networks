import os

import hydra
import jax
import jax.numpy as jnp
from flax.serialization import to_state_dict
from omegaconf import DictConfig, OmegaConf

from data import get_dataset
from models.jax import get_model
from neural_kernels.ntk import ntk_eigendecomposition


@hydra.main(config_path="config/compute_ntk", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    model_key = jax.random.PRNGKey(cfg.seed)

    train_ds, test_ds = get_dataset(**cfg.data)

    data = jnp.concatenate([train_ds["data"], test_ds["data"]], axis=0)

    model = get_model(**cfg.model)
    init_variables = model.init(model_key, jnp.zeros(cfg.shape, jnp.float32))

    print("Computing NTK at init...")
    (
        eigvals_init,
        eigvecs_init,
        _,
        _,
    ) = ntk_eigendecomposition(model, init_variables, data, **cfg.ntk)

    print("Done!")

    print("Saving results...")

    init_variables_state_dict = to_state_dict(init_variables)

    save_path = f"{hydra.utils.get_original_cwd()}/artifacts/eigenfunctions/{cfg.data.dataset}/{cfg.model.model_name}"
    os.makedirs(save_path, exist_ok=True)

    jnp.save(f"{save_path}/eigvecs.npy", eigvecs_init)
    jnp.save(f"{save_path}/eigvals.npy", eigvals_init)
    jnp.save(
        f"{save_path}/init_variables.npy",
        init_variables_state_dict,
    )
    jnp.save(f"{save_path}/data.npy", data)


if __name__ == "__main__":
    main()
