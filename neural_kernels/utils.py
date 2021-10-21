import jax.numpy as jnp


def flatten_kernel(K):
    return K.transpose([0, 2, 1, 3]).reshape([K.shape[0] * K.shape[2], K.shape[1] * K.shape[3]])


def unflatten_kernel(K, num_classes):
    return K.reshape(
        [K.shape[0] // num_classes, num_classes, K.shape[1] // num_classes, num_classes]
    ).transpose([0, 2, 1, 3])


def unflatten_kernel_eigvecs(eigvecs, num_classes):
    return eigvecs.reshape([-1, eigvecs.shape[1] // num_classes, num_classes])


def flatten_kernel_eigvecs(eigvecs):
    return eigvecs.reshape([-1, eigvecs.shape[1] * eigvecs.shape[2]])


def sort_nads(nads, eigvals):
    indices = jnp.flipud(jnp.argsort(jnp.abs(eigvals)))
    return nads[indices], eigvals[indices]
