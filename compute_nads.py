import os

import hydra
import jax
import jax.numpy as jnp
from flax.serialization import to_state_dict
from omegaconf import DictConfig, OmegaConf

from models.jax import get_model
from neural_kernels.nads import mixed_derivative_nad_decomposition
from utils.misc import get_apply_fn


@hydra.main(config_path="config/compute_nads", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # Load model
    model_key = jax.random.PRNGKey(cfg.seed)

    model = get_model(**cfg.model)
    init_variables = model.init(model_key, jnp.zeros(cfg.nads.shape, jnp.float32))

    apply_fn = get_apply_fn(model, expose_bn=False, variables=init_variables, train=False)
    _, init_params = init_variables.pop("params")

    print("Computing NADs...")
    # Compute NADs
    eigvals, nads = mixed_derivative_nad_decomposition(apply_fn, init_params, **cfg.nads)
    print("Done!")

    print("Saving results...")
    # Save results
    init_variables_state_dict = to_state_dict(init_variables)

    save_path = f"{hydra.utils.get_original_cwd()}/artifacts/nads/{cfg.model.model_name}"
    os.makedirs(save_path, exist_ok=True)

    jnp.save(f"{save_path}/nads.npy", nads)
    jnp.save(f"{save_path}/eigvals.npy", eigvals)
    jnp.save(
        f"{save_path}/init_variables.npy",
        init_variables_state_dict,
    )


if __name__ == "__main__":
    main()
