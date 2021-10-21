import jax
import jax.numpy as jnp
import numpy as onp
from jax import jit
from jax.flatten_util import ravel_pytree
from scipy.sparse.linalg import LinearOperator, svds


def mixed_derivative_nad_decomposition(
    apply_fn,
    params,
    shape,
    data=None,
    sigma=0.0,
    seed=0,
):
    if data is None:
        x = jax.random.normal(jax.random.PRNGKey(seed), shape, dtype=jnp.float32) * sigma
    else:
        print("Using provided data")
        x = data

    vec_params, unravel_pytree = ravel_pytree(params)

    @jit
    def mixed_derivative_vp(v):
        """Computes grad_x(J)v"""
        v = jnp.tile(v, (shape[0], 1)).reshape([-1] + shape[1:])

        def input_derivative(params):
            return jax.jvp(lambda x: apply_fn(params, x)[:, 0].sum() / shape[0], [x], [v])[1]

        return ravel_pytree(jax.grad(input_derivative)(params))[0]

    @jit
    def mixed_derivative_tvp(u):
        """Computes u^T J(grad_x(f))"""
        new_params = unravel_pytree(u)

        def weight_derivative(x):
            return jax.jvp(
                lambda params: apply_fn(params, x)[:, 0].sum() / shape[0], [params], [new_params]
            )[1]

        return jax.grad(weight_derivative)(x).sum(axis=0)

    A = LinearOperator(
        (len(vec_params), onp.prod(shape[1:])),
        matvec=mixed_derivative_vp,
        rmatvec=mixed_derivative_tvp,
    )
    _, sing_values, sing_vecs = svds(A, onp.prod(shape[1:]) - 1, return_singular_vectors="vh")

    ordered_indices = onp.argsort(sing_values)[::-1]
    sing_values = sing_values[ordered_indices]
    sing_vecs = sing_vecs[ordered_indices]

    return sing_values ** 2, sing_vecs
