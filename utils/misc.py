import jax
from flax.core.frozen_dict import freeze
from jax import numpy as jnp
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_map


def binarize_labels(labels, threshold_class):
    return (labels <= threshold_class).astype(int)


def binarize_eigenfunctions(eigenfunction):
    return ((jnp.sign(eigenfunction) + 1) // 2).astype(float)


def has_batchnorm(model_state):
    return "batch_stats" in model_state.unfreeze().keys()


# FIXME: Computes average distance over batch
def cos_dist(x1, x2):
    inner_prod = jnp.sum(x1 * x2, axis=-1)
    norm1 = jnp.linalg.norm(x1, axis=-1)
    norm2 = jnp.linalg.norm(x2, axis=-1)
    dist = 1 - (inner_prod / (norm1 * norm2))
    return jnp.mean(dist)


def kernel_cos_dist(K1, K2):
    inner_prod = jnp.trace(K1 @ K2.T)
    norm1 = jnp.sqrt(jnp.trace(K1 @ K1.T))
    norm2 = jnp.sqrt(jnp.trace(K2 @ K2.T))
    dist = 1 - (inner_prod / (norm1 * norm2))
    return dist


def mse_dist(preds, labels):
    def squared_error(pred, y):
        return jnp.inner(y - pred, y - pred)
        # We vectorize the previous to compute the average of the loss on all samples.

    return jnp.mean(jax.vmap(squared_error)(preds, labels), axis=0)


def params_cos_dist(params1, params2):
    theta1, _ = ravel_pytree(params1)
    theta2, _ = ravel_pytree(params2)
    inner_prod = jnp.sum(theta1 * theta2)
    norm1 = jnp.linalg.norm(theta1)
    norm2 = jnp.linalg.norm(theta2)
    dist = 1 - (inner_prod / (norm1 * norm2))
    return dist


def params_mse_dist(params1, params2):
    theta1, _ = ravel_pytree(params1)
    theta2, _ = ravel_pytree(params2)
    return ((theta1 - theta2) ** 2).sum()


def params_euclidean_dist(params1, params2):
    return jnp.sqrt(params_mse_dist(params1, params2))


def weight_norm(params):
    theta, _ = ravel_pytree(params)
    return jnp.linalg.norm(theta)


def pairwise_angles(V1, V2):
    return V1.T @ V2


def rayleigh_quotient(V, K):
    @jax.vmap
    def quotient(v):
        return (v.T @ K @ v) / jnp.linalg.norm(v)

    return quotient(V)


def make_variables(params, model_state):
    return freeze({"params": params, **model_state})


def weight_sum(params):
    theta, _ = ravel_pytree(params)
    return theta.sum()


def weight_energy(params):
    squared_params = tree_map(lambda p: p ** 2, params)
    return weight_sum(squared_params)


def vec_weight_energy(batch_params):
    batch_size = batch_params.shape[0] if batch_params.shape[0] > 0 else 1
    batch_params = batch_params.reshape([batch_size, -1])
    s = (batch_params ** 2).sum(axis=-1)
    return s


def distparams(init_params, end_params):
    dist_summary = {}
    dist_summary["params.cos"] = jax.device_get(params_cos_dist(init_params, end_params)).item()
    dist_summary["params.mse"] = jax.device_get(
        params_euclidean_dist(init_params, end_params)
    ).item()
    return dist_summary


def get_apply_fn(model, expose_bn=True, variables=None, train=False):
    if not expose_bn:
        if variables is None:
            raise TypeError("You must specify batch norm parameters")

        model_state, _ = variables.pop("params")

        def apply_fn(params, x):
            apply_vars = make_variables(params, model_state)
            logits = model.apply(apply_vars, x, mutable=False)
            return logits

    else:

        def apply_fn(variables, x):
            logits = model.apply(variables, x, mutable=train)
            return logits

    return apply_fn


def clip_norm(x, norm, epsilon):
    # Clipping to norm ball
    if norm not in [jnp.inf, 2]:
        raise ValueError("norm must be jnp.inf or 2.")

    axis = tuple(range(1, len(x.shape)))
    avoid_zero_div = 1e-12
    if norm == jnp.inf:
        x = jnp.clip(x, a_min=-epsilon, a_max=epsilon)
    elif norm == 2:
        # avoid_zero_div must go inside sqrt to avoid a divide by zero in the gradient through this operation
        norm = jnp.sqrt(
            jnp.maximum(avoid_zero_div, jnp.sum(jnp.square(x), axis=axis, keepdims=True))
        )
        # We must *clip* to within the norm ball, not *normalize* onto the surface of the ball
        factor = jnp.minimum(1.0, jnp.divide(epsilon, norm))
        x = x * factor
    return x
