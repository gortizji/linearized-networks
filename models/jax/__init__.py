import copy
from functools import partial

import flax.linen as nn
from omegaconf.omegaconf import OmegaConf

from .lenet import LeNet
from .mlp import MLP
from .resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152, ResNet200


def get_model(model_name, model_config):
    model_dict = _substitute_activation(model_config)

    if model_name in globals():
        model = globals()[model_name](**model_dict)
    else:
        raise ValueError("Specified model not accepted")

    return model


def _substitute_activation(model_config):
    model_dict = OmegaConf.to_container(model_config)

    if "activation" in model_dict.keys():
        value = model_dict["activation"]
        if value == "ReLU":
            model_dict["activation"] = nn.relu
        elif value == "GeLU":
            model_dict["activation"] = nn.gelu
        elif value == "sigmoid":
            model_dict["activation"] = nn.sigmoid
        elif value == "tanh":
            model_dict["activation"] = nn.tanh
        elif value == "softplus":
            model_dict["activation"] = nn.softplus
        else:
            raise ValueError(f"{value} is not a valid activation function.")

    return model_dict
