import hydra
import jax
import jax.numpy as jnp
from omegaconf import DictConfig, OmegaConf

from data import get_samples_from_dataset
from models.jax import get_model
from train import sgd_train
from utils.misc import binarize_labels, make_variables, params_mse_dist


def construct_data(num_train, num_test, seed):
    train_data, train_labels = get_samples_from_dataset(
        num_train, "cifar10", normalize=True, train=True, seed=seed
    )
    test_data, test_labels = get_samples_from_dataset(
        num_test, "cifar10", normalize=True, train=False, seed=seed
    )
    train_labels = binarize_labels(train_labels, 4)
    test_labels = binarize_labels(test_labels, 4)

    train_ds = {"data": train_data, "labels": train_labels}
    test_ds = {"data": test_data, "labels": test_labels}
    return train_ds, test_ds


@hydra.main(config_path="config/binary-cifar", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    train_key, model_key = jax.random.split(jax.random.PRNGKey(cfg.seed))

    # Generate training data
    train_ds, test_ds = construct_data(cfg.num_train, cfg.num_test, cfg.seed + 999)

    # Initialize model
    model = get_model(**cfg.model)
    init_variables = model.init(model_key, jnp.zeros((1, 32, 32, 3), jnp.float32))

    # Train model
    output_summary = sgd_train(
        train_ds=train_ds,
        test_ds=test_ds,
        apply_fn=model.apply,
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
