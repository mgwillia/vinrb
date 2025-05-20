### code adapted from https://github.com/abhyantrika/mediainr ###

import torch
import torch.nn as nn
import torch.nn.functional as F


class Sine(nn.Module):
    """Sine activation with scaling.

    Args:
        w0 (float): Omega_0 parameter from SIREN paper.
    """
    def __init__(self, w0:int=30.,c:int=6):
        super().__init__()
        self.w0 = w0
        self.c = c #std of the uniform distribution

    def forward(self, x):
        return torch.sin(self.w0 * x)


def get_activation(activation:str, **kwargs):
    """
    Returns the activation function based on the provided name.

    :param name: Name of the activation function.
    :return: Corresponding activation function.
    
    """

    if (activation == 'none') or (activation == 'linear') or (activation is None):
        return nn.Identity()

    elif activation.lower() == 'relu':
        return nn.ReLU()
    elif activation.lower() == 'leakyrelu':
        return nn.LeakyReLU()
    elif activation.lower() == 'tanh':
        return nn.Tanh()
    elif activation.lower() == 'sigmoid':
        return nn.Sigmoid()
    elif activation.lower() == 'gelu':
        return nn.GELU()
    elif activation.lower() == 'sine':
        return Sine(kwargs.get('w0', 30.0),kwargs.get('c',6.0))
    else:
        raise ValueError('Unknown activation function {}'.format(activation))

