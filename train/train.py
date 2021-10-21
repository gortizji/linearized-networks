import copy

import jax
import numpy as onp
import optax
from jax import random

from utils.metrics import compute_binary_accuracy_metrics
from utils.misc import make_variables, weight_norm

from .utils import (
    create_learning_rate_fn,
    create_train_state,
    generate_binary_cross_entropy_loss_fn,
    print_progress,
    split_batch_indices,
)


def create_train_step_fn(
    apply_fn,
    optimizer,
):
    @jax.jit
    def train_step_fn(state, batch, rng_key):

        loss_fn = generate_binary_cross_entropy_loss_fn(apply_fn, state, batch)

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        aux, grad = grad_fn(state.target)

        new_model_state, logits = aux[1]
        updates, new_opt_state = optimizer.update(grad, state.opt_state, params=state.target)
        new_target = optax.apply_updates(state.target, updates)

        metrics = compute_binary_accuracy_metrics(logits, batch["labels"])

        metrics["loss_grad_norm"] = jax.device_get(weight_norm(grad))
        metrics["weight_norm"] = jax.device_get(weight_norm(new_target))

        new_state = state.replace(
            step=state.step + 1,
            model_state=new_model_state,
            opt_state=new_opt_state,
            target=new_target,
        )
        return new_state, metrics

    return train_step_fn


def create_eval_model_fn(apply_fn):
    @jax.jit
    def eval_step_fn(state, batch):

        params = state.target
        variables = make_variables(params, state.model_state)
        logits = apply_fn(variables, batch["data"], mutable=False)

        metrics = compute_binary_accuracy_metrics(logits, batch["labels"])

        return metrics

    def eval_model_fn(state, test_ds, batch_size=-1):
        indices, _ = split_batch_indices(batch_size, None, test_ds)
        batch_metrics = []
        for _, perm in enumerate(indices):
            batch = {k: v[perm] for k, v in test_ds.items()}
            metrics = eval_step_fn(state, batch)
            batch_metrics.append(jax.device_get(metrics))

        metrics_np = {
            k: onp.mean([metrics[k] for metrics in batch_metrics]) for k in batch_metrics[0]
        }

        return metrics_np

    return eval_model_fn


def sgd_train(
    train_ds,
    test_ds,
    epochs,
    max_lr,
    batch_size,
    lr_schedule,
    momentum,
    apply_fn,
    init_variables,
    key,
    optimizer="sgd",
):
    print("Initializing optimizer...")

    learning_rate_fn = create_learning_rate_fn(
        base_learning_rate=max_lr,
        num_epochs=epochs,
        steps_per_epoch=len(train_ds["labels"]) // batch_size,
        lr_schedule=lr_schedule,
    )

    init_state, optimizer = create_train_state(
        optimizer_type=optimizer,
        momentum=momentum,
        init_variables=init_variables,
        learning_rate_fn=learning_rate_fn,
    )
    state = copy.deepcopy(init_state)

    train_step_fn = create_train_step_fn(
        apply_fn=apply_fn,
        optimizer=optimizer,
    )
    eval_model_fn = create_eval_model_fn(
        apply_fn=apply_fn,
    )

    print("Starting training...")
    rng = random.split(key, epochs)

    for epoch in range(epochs):
        perms, steps_per_epoch = split_batch_indices(batch_size, rng[epoch], train_ds)
        for step, perm in enumerate(perms):
            batch = {k: v[perm] for k, v in train_ds.items()}

            state, metrics = train_step_fn(
                state=state, batch=batch, rng_key=jax.random.fold_in(rng[epoch], step)
            )

        train_summary = jax.tree_map(lambda x: jax.device_get(x.mean()), metrics)
        test_summary = eval_model_fn(state, test_ds, batch_size)

        print_progress(epoch, test_summary, train_summary)

    output_summary = {
        "train": train_summary,
        "test": test_summary,
        "init_state": init_state,
        "end_state": state,
    }

    return output_summary
