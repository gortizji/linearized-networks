import jax
import jax.numpy as jnp
import numpy as onp
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split

from .utils import normalize_data


def get_dataset(dataset, normalize=False, data_dir=None):
    """Load dataset train and test datasets into memory."""
    ds_builder = tfds.builder(dataset, data_dir=data_dir)
    ds_builder.download_and_prepare()

    train_data, train_labels = tfds.as_numpy(
        ds_builder.as_dataset(split="train", batch_size=-1, as_supervised=True, shuffle_files=False)
    )
    train_data = jnp.float32(train_data) / 255.0
    if normalize:
        train_data, mean, std = normalize_data(train_data)

    train_ds = {"data": train_data, "labels": train_labels}

    test_data, test_labels = tfds.as_numpy(
        ds_builder.as_dataset(split="test", batch_size=-1, as_supervised=True)
    )
    test_data = jnp.float32(test_data) / 255.0
    if normalize:
        test_data, _, _ = normalize_data(test_data, mean, std)

    test_ds = {"data": test_data, "labels": test_labels}

    return train_ds, test_ds


def get_samples_from_dataset(num_samples, dataset, train=True, normalize=False, seed=4242):
    train_ds, test_ds = get_dataset(dataset, normalize)
    num_train = train_ds["data"].shape[0]
    num_test = test_ds["data"].shape[0]

    if train:
        if num_samples < num_train:
            data, _, labels, _ = train_test_split(
                train_ds["data"],
                train_ds["labels"],
                train_size=num_samples,
                stratify=train_ds["labels"],
                random_state=seed,
            )
        else:
            data = train_ds["data"]
            labels = train_ds["labels"]
    else:
        if num_samples < num_test:
            data, _, labels, _ = train_test_split(
                test_ds["data"],
                test_ds["labels"],
                train_size=num_samples,
                stratify=test_ds["labels"],
                random_state=seed,
            )
        else:
            data = test_ds["data"]
            labels = test_ds["labels"]

    return data, labels


def linearly_separable_dataset(direction, epsilon, sigma, shape, num_train, num_test, rng_key):
    noise_key, labels_key = jax.random.split(rng_key)

    w = sigma * jax.random.normal(
        noise_key,
        [
            num_train + num_test,
        ]
        + list(shape),
    )
    w_proj = jnp.einsum("bhwc, hwc->b", w, direction)
    w = w - w_proj[:, None, None, None] * direction[None, ...]

    labels = jax.random.bernoulli(
        labels_key,
        p=0.5,
        shape=[
            num_train + num_test,
        ],
    ).astype(int)
    y = 2 * labels - 1
    x = (epsilon / 2) * y[:, None, None, None] * direction[None, :, :, :] + w

    train_ds = {"data": x[:num_train], "labels": labels[:num_train]}
    test_ds = {"data": x[num_train:], "labels": labels[num_train:]}
    return train_ds, test_ds
