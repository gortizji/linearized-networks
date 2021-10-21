import hydra
import jax
import jax.numpy as jnp
from einops import rearrange
from flax.serialization import from_state_dict
from omegaconf import DictConfig, OmegaConf

from data import linearly_separable_dataset
from models.jax import get_model
from neural_kernels.ntk import linearize_model
from train import sgd_train
from utils.misc import make_variables, params_mse_dist


def load_dataset_and_model(cfg, data_key):

    save_path = f"{hydra.utils.get_original_cwd()}/artifacts/nads/{cfg.model.model_name}"

    nads = jnp.load(f"{save_path}/nads.npy", allow_pickle=True)

    train_nad = rearrange(
        nads[cfg.label_idx],
        "(h w c) -> h w c",
        c=cfg.data.shape[2],
        h=cfg.data.shape[0],
        w=cfg.data.shape[1],
    )
    train_ds, test_ds = linearly_separable_dataset(
        direction=train_nad, rng_key=data_key, **cfg.data
    )

    model = get_model(**cfg.model)
    init_variables_state_dict = jnp.load(f"{save_path}/init_variables.npy", allow_pickle=True)[()]
    init_variables = model.init(
        jax.random.PRNGKey(0),
        jnp.zeros(
            [
                1,
            ]
            + list(cfg.data.shape)
        ),
    )

    loaded_variables = from_state_dict(init_variables, init_variables_state_dict)

    return train_ds, test_ds, model, loaded_variables, nads


@hydra.main(config_path="config/train_nads", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    main_key = jax.random.PRNGKey(cfg.seed)
    train_key, data_key = jax.random.split(main_key, 2)

    train_ds, test_ds, model, init_variables, nads = load_dataset_and_model(cfg, data_key)

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


if __name__ == "__main__":
    main()
