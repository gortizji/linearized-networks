import functools
from typing import Callable, Sequence, Union

import flax.linen as nn


class LeNet(nn.Module):
    activation: Union[None, Callable] = nn.relu
    kernel_size: Sequence[int] = (5, 5)
    strides: Sequence[int] = (2, 2)
    window_shape: Sequence[int] = (2, 2)
    num_classes: int = 1
    features: Sequence[int] = (6, 16, 120, 84, 1)
    pooling: bool = True
    padding: str = "SAME"

    @nn.compact
    def __call__(self, x):
        conv = functools.partial(nn.Conv, padding=self.padding)
        x = conv(features=self.features[0], kernel_size=tuple(self.kernel_size))(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.pooling:
            x = nn.avg_pool(x, window_shape=tuple(self.window_shape), strides=tuple(self.strides))

        x = conv(features=self.features[1], kernel_size=tuple(self.kernel_size))(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.pooling:
            x = nn.avg_pool(x, window_shape=tuple(self.window_shape), strides=tuple(self.strides))

        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(self.features[2])(x)
        if self.activation is not None:
            x = self.activation(x)
        x = nn.Dense(self.features[3])(x)
        if self.activation is not None:
            x = self.activation(x)

        x = nn.Dense(self.num_classes)(x)
        # x = nn.log_softmax(x)
        return x
