import jax.numpy as jnp


def normalize_data(data, mean=None, std=None):
    if mean is None or std is None:
        mean = jnp.mean(data, axis=[0, 1, 2])[jnp.newaxis, jnp.newaxis, jnp.newaxis, :]
        std = jnp.std(data, axis=[0, 1, 2])[jnp.newaxis, jnp.newaxis, jnp.newaxis, :]

    data = data - mean
    data = data / std
    return data, mean, std


def normalize_dataset(ds, mean=None, std=None):
    data = ds["data"]
    data, mean, std = normalize_data(data, mean, std)
    ds["data"] = data
    return ds, mean, std
