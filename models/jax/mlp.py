from typing import Callable, Sequence

import flax.linen as nn


class MLP(nn.Module):
    features: Sequence[int]
    activation: Callable = nn.relu

    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))
        for feat in self.features[:-1]:
            x = self.activation(nn.Dense(feat)(x))
        x = nn.Dense(self.features[-1])(x)
        # x = nn.log_softmax(x)
        return x
