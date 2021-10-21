import chex
import jax
import jax.numpy as jnp
import optax

from utils.loss import binary_cross_entropy_loss_with_logits
from utils.misc import make_variables


@chex.dataclass
class TrainState:
    step: int
    opt_state: optax.OptState
    target: chex.ArrayTree
    model_state: chex.ArrayTree


def create_train_state(
    init_variables,
    optimizer_type,
    learning_rate_fn,
    momentum=None,
):
    """Create initial training state."""
    model_state, params = init_variables.pop("params")
    transforms = []

    if optimizer_type == "sgd":
        transforms += [optax.sgd(learning_rate=learning_rate_fn, momentum=momentum)]
    elif optimizer_type == "adam":
        transforms += [optax.adam(learning_rate=learning_rate_fn)]
    else:
        raise ValueError("Optimizer type not accepted.")

    optimizer = optax.chain(*transforms)
    opt_state = optimizer.init(params)

    state = TrainState(step=0, opt_state=opt_state, target=params, model_state=model_state)
    return state, optimizer


def generate_binary_cross_entropy_loss_fn(apply_fn, state, batch):
    def loss_fn(params):
        variables = make_variables(params, state.model_state)
        logits, new_model_state = apply_fn(variables, batch["data"], mutable=["batch_stats"])
        loss = binary_cross_entropy_loss_with_logits(logits, batch["labels"])
        return loss, (new_model_state, logits)

    return loss_fn


def create_learning_rate_fn(base_learning_rate, steps_per_epoch, num_epochs, lr_schedule="cyclic"):
    if lr_schedule == "linear":

        schedule_fn = optax.linear_schedule(
            init_value=base_learning_rate,
            end_value=0,
            transition_steps=num_epochs * steps_per_epoch,
        )

    elif lr_schedule == "fixed":

        schedule_fn = optax.constant_schedule(value=base_learning_rate)

    elif lr_schedule == "piecewise":
        raise NotImplementedError()

    elif lr_schedule == "cyclic":

        schedule_fn = optax.linear_onecycle_schedule(
            peak_value=base_learning_rate,
            transition_steps=num_epochs * steps_per_epoch,
            pct_start=0.4,
            pct_final=1,
            div_factor=1e8,
            final_div_factor=1e8,
        )

    else:
        raise ValueError("Learning rate error not specified")

    return schedule_fn


def split_batch_indices(batch_size, rng, ds):
    ds_size = len(ds["data"])
    if batch_size > 0:
        steps_per_epoch = ds_size // batch_size
    else:
        steps_per_epoch = 1

    if rng is not None:
        indices = jax.random.permutation(rng, ds_size)
    else:
        indices = jnp.arange(ds_size)

    if batch_size > 0:
        indices = indices[: steps_per_epoch * batch_size]  # skip incomplete batch
    indices = indices.reshape((steps_per_epoch, batch_size))
    return indices, steps_per_epoch


def print_progress(epoch, test_summary, train_summary):
    print(
        "Epoch: %d, train loss: %.4f, train acc.: %.2f%%, test acc.: %.2f%%"
        % (
            epoch,
            train_summary["loss"],
            train_summary["accuracy"] * 100,
            test_summary["accuracy"] * 100,
        )
    )
