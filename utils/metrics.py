import jax.numpy as jnp

from .loss import binary_cross_entropy_loss_with_logits, mse_loss, softmax_cross_entropy_loss


def compute_accuracy_metrics(logits, labels):
    loss = softmax_cross_entropy_loss(logits, labels)
    if len(labels.shape) > 1:
        accuracy = jnp.mean(jnp.argmax(logits, -1) == jnp.argmax(labels, -1))
    else:
        accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    metrics = {
        "loss": loss,
        "accuracy": accuracy,
    }
    return metrics


def compute_binary_accuracy_metrics(logits, labels):
    loss = binary_cross_entropy_loss_with_logits(logits, labels)
    predictions = ((jnp.sign(logits) + 1) // 2).astype(float)
    labels_bin = (labels[:, jnp.newaxis] > 0.5).astype(float)
    accuracy = 1 - jnp.abs(predictions - labels_bin).mean()
    metrics = {
        "loss": loss,
        "accuracy": accuracy,
    }
    return metrics


def compute_mse_metrics(logits, targets):
    loss = mse_loss(logits, targets)
    metrics = {"loss": loss}
    return metrics
