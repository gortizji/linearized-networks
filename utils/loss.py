import jax
import jax.numpy as jnp
import optax


def softmax_cross_entropy_loss(logits, labels):
    if len(labels.shape) <= 1:
        num_classes = logits.shape[-1]
        soft_labels = jax.nn.one_hot(labels, num_classes=num_classes)
    else:
        soft_labels = labels
    return jnp.mean(optax.softmax_cross_entropy(logits, soft_labels))


def binary_cross_entropy_loss_with_logits(logits, labels):
    return jnp.mean(optax.sigmoid_binary_cross_entropy(logits, labels[:, jnp.newaxis]))


def mse_loss(preds, labels):
    return jnp.mean(optax.l2_loss(preds, labels))
