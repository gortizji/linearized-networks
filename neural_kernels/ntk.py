from functools import partial

import jax
import jax.numpy as jnp
import neural_tangents as nt
import numpy as np
from jax.tree_util import tree_leaves, tree_map, tree_multimap
from scipy.sparse.linalg import eigsh

from neural_kernels.utils import flatten_kernel
from train.utils import split_batch_indices
from utils.misc import get_apply_fn, vec_weight_energy


def linearize_model(model, variables_0):
    """Linearize dynamics, i.e., NTK with bias"""
    model_state, params_0 = variables_0.pop("params")

    original_apply_fn = get_apply_fn(model, expose_bn=False, variables=variables_0, train=False)

    new_apply_fn = nt.linearize(original_apply_fn, params_0)

    def apply_lin_fn(variables, x, train=True, mutable=False):
        if not train or not mutable:
            return new_apply_fn(variables["params"], x)
        else:
            return new_apply_fn(variables["params"], x), model_state

    return apply_lin_fn


def linearize_diff_model(model, variables_0):
    """Differential linearize dynamics, i.e., NTK w/o bias"""
    model_state, params_0 = variables_0.pop("params")

    original_apply_fn = get_apply_fn(model, expose_bn=False, variables=variables_0, train=False)

    new_apply_fn = nt.linearize(original_apply_fn, params_0)

    def apply_lin_fn(variables, x, mutable=False):
        if not mutable:
            return new_apply_fn(variables["params"], x) - original_apply_fn(params_0, x)
        else:
            return (
                new_apply_fn(variables["params"], x) - original_apply_fn(params_0, x),
                model_state,
            )

    return apply_lin_fn


def get_ntk_fn(model, variables, batch_size):
    apply_fn = get_apply_fn(model, expose_bn=False, variables=variables, train=False)
    kernel_fn = nt.batch(
        nt.empirical_kernel_fn(apply_fn, vmap_axes=0, implementation=1, trace_axes=()),
        batch_size=batch_size,
        device_count=-1,
        store_on_device=False,
    )

    def expanded_kernel_fn(data1, data2, kernel_type, params):
        K = kernel_fn(data1, data2, kernel_type, params)
        return flatten_kernel(K)

    return expanded_kernel_fn


def ntk_eigendecomposition(model, variables, data, batch_size, nystrom_dims=None, num_eigvecs=1000):
    kernel_fn = get_ntk_fn(model, variables, batch_size)

    if nystrom_dims is not None and nystrom_dims < data.shape[0]:
        eigvals, eigvecs, eigvals_m, eigvecs_m = nystrom_eigendecomposition(
            kernel_fn, variables["params"], data, nystrom_dims
        )

        return eigvals, eigvecs, eigvals_m, eigvecs_m

    else:
        ntk_matrix = kernel_fn(data, None, "ntk", variables["params"])
        eigvals, eigvecs = eigsh(jax.device_get(ntk_matrix), k=num_eigvecs)
        eigvals = np.flipud(eigvals)
        eigvecs = np.flipud(eigvecs.T)

        return eigvals, eigvecs, eigvals, eigvecs


def nystrom_eigendecomposition(kernel_fn, params, data, nystrom_dims):
    n = data.shape[0]

    # Split data
    Xm = data[:nystrom_dims]
    Xrest = data[nystrom_dims:]

    # Compute small gram matrices
    Km = kernel_fn(Xm, None, "ntk", params)
    Krest = kernel_fn(Xrest, Xm, "ntk", params)

    Knystrom = np.concatenate([Km, Krest], axis=0)

    # Approximate eigenvectors
    eigvals_m, eigvecs_m = eigsh(jax.device_get(Km), k=nystrom_dims - 1)
    eigvals_m = np.flipud(eigvals_m)
    eigvecs_m = np.flipud(eigvecs_m.T)

    eigvals_approx = (n / nystrom_dims) * eigvals_m
    eigvecs_approx = np.sqrt(nystrom_dims / n) * Knystrom @ (eigvecs_m.T * (1 / eigvals_m))

    return eigvals_approx, eigvecs_approx.T, eigvals_m, eigvecs_m


def get_ntk_alignment_fn(model, variables, batch_size, eig_batch_size=-1):

    apply_fn = get_apply_fn(model=model, expose_bn=False, variables=variables, train=False)

    @jax.jit
    def ntk_alignment(x, y, params):
        jac = jax.jacrev(partial(apply_fn, x=x))(params)
        vjp = tree_map(lambda p: jnp.einsum("vbi, bi...->v...", y, p, optimize=True), jac)
        return vjp

    def ntk_alignment_fn(data, labels, params):
        batch_indices, _ = split_batch_indices(batch_size, None, {"data": data})
        eig_batch_indices, _ = split_batch_indices(eig_batch_size, None, {"data": labels})
        vec_alignments = []
        for eig_batch in eig_batch_indices:
            batch_labels = labels[eig_batch]
            for n, batch in enumerate(batch_indices):
                if n == 0:
                    vjp = ntk_alignment(data[batch], batch_labels[:, batch, :], params)
                else:
                    vjp_ = ntk_alignment(data[batch], batch_labels[:, batch, :], params)
                    vjp = tree_multimap(lambda p1, p2: p1 + p2, vjp_, vjp)

            array_energies = [vec_weight_energy(p) for p in tree_leaves(vjp)]
            vec_alignment = jnp.sum(jnp.array(array_energies), axis=0)

            vec_alignments.append(vec_alignment / (vec_weight_energy(batch_labels)))

        return jnp.concatenate(vec_alignments, axis=0)

    return ntk_alignment_fn
