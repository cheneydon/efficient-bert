import torch
import torch.nn.functional as F
import logging
import models
from deap import gp


class Operator(object):
    def __init__(self, sel_min=True):
        self.eps = 1e-5
        self.sel_min = sel_min  # Use when intermediate dim and input dim are mismatch

    def linear(self, x, y):
        if isinstance(x, torch.Tensor) and isinstance(y, list):
            return torch.matmul(x, y[0]) + y[1]
        else:
            raise KeyError('Input of \'linear\' is invalid')

    def linear1_1(self, x, y):
        if isinstance(x, torch.Tensor) and isinstance(y, list):
            return torch.matmul(x, y[0]) + y[1]
        else:
            raise KeyError('Input of \'linear1\' is invalid')

    def linear1_2(self, x, y):
        if isinstance(x, torch.Tensor) and isinstance(y, list):
            return torch.matmul(x, y[0]) + y[1]
        else:
            raise KeyError('Input of \'linear1\' is invalid')

    def linear1_3(self, x, y):
        if isinstance(x, torch.Tensor) and isinstance(y, list):
            return torch.matmul(x, y[0]) + y[1]
        else:
            raise KeyError('Input of \'linear1\' is invalid')

    def linear1_4(self, x, y):
        if isinstance(x, torch.Tensor) and isinstance(y, list):
            return torch.matmul(x, y[0]) + y[1]
        else:
            raise KeyError('Input of \'linear1\' is invalid')

    def linear2_1(self, x, y):
        if isinstance(x, torch.Tensor) and isinstance(y, list):
            return torch.matmul(x, y[0]) + y[1]
        else:
            raise KeyError('Input of \'linear2\' is invalid')

    def linear2_2(self, x, y):
        if isinstance(x, torch.Tensor) and isinstance(y, list):
            return torch.matmul(x, y[0]) + y[1]
        else:
            raise KeyError('Input of \'linear2\' is invalid')

    def linear2_3(self, x, y):
        if isinstance(x, torch.Tensor) and isinstance(y, list):
            return torch.matmul(x, y[0]) + y[1]
        else:
            raise KeyError('Input of \'linear2\' is invalid')

    def linear2_4(self, x, y):
        if isinstance(x, torch.Tensor) and isinstance(y, list):
            return torch.matmul(x, y[0]) + y[1]
        else:
            raise KeyError('Input of \'linear2\' is invalid')

    def linear3_1(self, x, y):
        if isinstance(x, torch.Tensor) and isinstance(y, list):
            return torch.matmul(x, y[0]) + y[1]
        else:
            raise KeyError('Input of \'linear3\' is invalid')

    def linear3_2(self, x, y):
        if isinstance(x, torch.Tensor) and isinstance(y, list):
            return torch.matmul(x, y[0]) + y[1]
        else:
            raise KeyError('Input of \'linear3\' is invalid')

    def linear3_3(self, x, y):
        if isinstance(x, torch.Tensor) and isinstance(y, list):
            return torch.matmul(x, y[0]) + y[1]
        else:
            raise KeyError('Input of \'linear3\' is invalid')

    def linear3_4(self, x, y):
        if isinstance(x, torch.Tensor) and isinstance(y, list):
            return torch.matmul(x, y[0]) + y[1]
        else:
            raise KeyError('Input of \'linear3\' is invalid')

    def linear4_1(self, x, y):
        if isinstance(x, torch.Tensor) and isinstance(y, list):
            return torch.matmul(x, y[0]) + y[1]
        else:
            raise KeyError('Input of \'linear4\' is invalid')

    def linear4_2(self, x, y):
        if isinstance(x, torch.Tensor) and isinstance(y, list):
            return torch.matmul(x, y[0]) + y[1]
        else:
            raise KeyError('Input of \'linear4\' is invalid')

    def linear4_3(self, x, y):
        if isinstance(x, torch.Tensor) and isinstance(y, list):
            return torch.matmul(x, y[0]) + y[1]
        else:
            raise KeyError('Input of \'linear4\' is invalid')

    def linear4_4(self, x, y):
        if isinstance(x, torch.Tensor) and isinstance(y, list):
            return torch.matmul(x, y[0]) + y[1]
        else:
            raise KeyError('Input of \'linear4\' is invalid')

    def add(self, x, y):
        if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
            min_channel = min(x.size(-1), y.size(-1))
            max_channel = max(x.size(-1), y.size(-1))
            if self.sel_min:
                x, y = x[:, :, :min_channel], y[:, :, :min_channel]
                return x + y
            else:  # select max
                if x.size(-1) == min_channel:
                    min_tensor, max_tensor = x, y
                else:
                    min_tensor, max_tensor = y, x
                pad_tensor = torch.zeros((min_tensor.size(0), min_tensor.size(1), max_channel - min_channel), device=min_tensor.device)
                min_tensor = torch.cat((min_tensor, pad_tensor), dim=-1)
                return max_tensor + min_tensor
        else:
            raise KeyError('Input of \'add\' is invalid')

    def mul(self, x, y):
        if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
            min_channel = min(x.size(-1), y.size(-1))
            max_channel = max(x.size(-1), y.size(-1))
            if self.sel_min:
                x, y = x[:, :, :min_channel], y[:, :, :min_channel]
                return x * y
            else:  # select max
                if x.size(-1) == min_channel:
                    min_tensor, max_tensor = x, y
                else:
                    min_tensor, max_tensor = y, x
                pad_tensor = torch.ones((min_tensor.size(0), min_tensor.size(1), max_channel - min_channel), device=min_tensor.device)
                min_tensor = torch.cat((min_tensor, pad_tensor), dim=-1)
                return max_tensor * min_tensor
        else:
            raise KeyError('Input of \'mul\' is invalid')

    def max(self, x, y):
        if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
            min_channel = min(x.size(-1), y.size(-1))
            max_channel = max(x.size(-1), y.size(-1))
            if self.sel_min:
                x, y = x[:, :, :min_channel], y[:, :, :min_channel]
                return torch.max(x, y)
            else:  # select max
                if x.size(-1) == min_channel:
                    min_tensor, max_tensor = x, y
                else:
                    min_tensor, max_tensor = y, x
                pad_tensor = torch.ones((min_tensor.size(0), min_tensor.size(1), max_channel - min_channel), device=min_tensor.device) / -1e10
                min_tensor = torch.cat((min_tensor, pad_tensor), dim=-1)
                return torch.max(max_tensor, min_tensor)
        else:
            raise KeyError('Input of \'max\' is invalid')

    def gelu(self, x):
        if isinstance(x, torch.Tensor):
            return models.gelu(x)
        else:
            raise KeyError('Input of \'gelu\' is invalid')

    def sigmoid(self, x):
        if isinstance(x, torch.Tensor):
            return torch.sigmoid(x)
        else:
            raise KeyError('Input of \'sigmoid\' is invalid')

    def tanh(self, x):
        if isinstance(x, torch.Tensor):
            return torch.tanh(x)
        else:
            raise KeyError('Input of \'tanh\' is invalid')

    def relu(self, x):
        if isinstance(x, torch.Tensor):
            return F.relu(x)
        else:
            raise KeyError('Input of \'relu\' is invalid')

    def leaky_relu(self, x):
        if isinstance(x, torch.Tensor):
            return F.leaky_relu(x)
        else:
            raise KeyError('Input of \'leaky_relu\' is invalid')

    def elu(self, x):
        if isinstance(x, torch.Tensor):
            return F.elu(x)
        else:
            raise KeyError('Input of \'elu\' is invalid')

    def swish(self, x):
        if isinstance(x, torch.Tensor):
            return x * torch.sigmoid(x)
        else:
            raise KeyError('Input of \'swish\' is invalid')


def register_custom_ops():  # For search stage 1
    custom_ops = Operator()
    pset = gp.PrimitiveSet('Main', 3)
    pset.addPrimitive(custom_ops.linear1_1, 2)
    pset.addPrimitive(custom_ops.linear1_2, 2)
    pset.addPrimitive(custom_ops.linear1_3, 2)
    pset.addPrimitive(custom_ops.linear1_4, 2)
    pset.addPrimitive(custom_ops.linear2_1, 2)
    pset.addPrimitive(custom_ops.linear2_2, 2)
    pset.addPrimitive(custom_ops.linear2_3, 2)
    pset.addPrimitive(custom_ops.linear2_4, 2)
    pset.addPrimitive(custom_ops.linear3_1, 2)
    pset.addPrimitive(custom_ops.linear3_2, 2)
    pset.addPrimitive(custom_ops.linear3_3, 2)
    pset.addPrimitive(custom_ops.linear3_4, 2)
    pset.addPrimitive(custom_ops.linear4_1, 2)
    pset.addPrimitive(custom_ops.linear4_2, 2)
    pset.addPrimitive(custom_ops.linear4_3, 2)
    pset.addPrimitive(custom_ops.linear4_4, 2)
    pset.addPrimitive(custom_ops.add, 2)
    pset.addPrimitive(custom_ops.mul, 2)
    pset.addPrimitive(custom_ops.max, 2)
    pset.addPrimitive(custom_ops.gelu, 1)
    pset.addPrimitive(custom_ops.sigmoid, 1)
    pset.addPrimitive(custom_ops.tanh, 1)
    pset.addPrimitive(custom_ops.relu, 1)
    pset.addPrimitive(custom_ops.leaky_relu, 1)
    pset.addPrimitive(custom_ops.elu, 1)
    pset.addPrimitive(custom_ops.swish, 1)
    pset.renameArguments(ARG0='x', ARG1='wb1', ARG2='wb2')
    pset.activations = ['gelu', 'sigmoid', 'tanh', 'relu', 'leaky_relu', 'elu', 'swish']
    pset.all_linear_ids = [
        '1_1', '1_2', '1_3', '1_4',
        '2_1', '2_2', '2_3', '2_4',
        '3_1', '3_2', '3_3', '3_4',
        '4_1', '4_2', '4_3', '4_4',
    ]
    return pset


def register_custom_ops2():  # For search stage 2
    custom_ops = Operator()
    pset = gp.PrimitiveSet('Main', 3)
    pset.addPrimitive(custom_ops.linear, 2)
    pset.addPrimitive(custom_ops.add, 2)
    pset.addPrimitive(custom_ops.mul, 2)
    pset.addPrimitive(custom_ops.max, 2)
    pset.addPrimitive(custom_ops.gelu, 1)
    pset.addPrimitive(custom_ops.sigmoid, 1)
    pset.addPrimitive(custom_ops.tanh, 1)
    pset.addPrimitive(custom_ops.relu, 1)
    pset.addPrimitive(custom_ops.leaky_relu, 1)
    pset.addPrimitive(custom_ops.elu, 1)
    pset.addPrimitive(custom_ops.swish, 1)
    pset.renameArguments(ARG0='x', ARG1='wb1', ARG2='wb2')
    pset.activations = ['gelu', 'sigmoid', 'tanh', 'relu', 'leaky_relu', 'elu', 'swish']
    return pset


def register_custom_ops3():  # For AutoTinyBERT
    custom_ops = Operator(sel_min=False)
    pset = gp.PrimitiveSet('Main', 3)
    pset.addPrimitive(custom_ops.linear1_1, 2)
    pset.addPrimitive(custom_ops.linear1_2, 2)
    pset.addPrimitive(custom_ops.linear1_3, 2)
    pset.addPrimitive(custom_ops.linear1_4, 2)
    pset.addPrimitive(custom_ops.linear2_1, 2)
    pset.addPrimitive(custom_ops.linear2_2, 2)
    pset.addPrimitive(custom_ops.linear2_3, 2)
    pset.addPrimitive(custom_ops.linear2_4, 2)
    pset.addPrimitive(custom_ops.linear3_1, 2)
    pset.addPrimitive(custom_ops.linear3_2, 2)
    pset.addPrimitive(custom_ops.linear3_3, 2)
    pset.addPrimitive(custom_ops.linear3_4, 2)
    pset.addPrimitive(custom_ops.linear4_1, 2)
    pset.addPrimitive(custom_ops.linear4_2, 2)
    pset.addPrimitive(custom_ops.linear4_3, 2)
    pset.addPrimitive(custom_ops.linear4_4, 2)
    pset.addPrimitive(custom_ops.add, 2)
    pset.addPrimitive(custom_ops.mul, 2)
    pset.addPrimitive(custom_ops.max, 2)
    pset.addPrimitive(custom_ops.gelu, 1)
    pset.addPrimitive(custom_ops.sigmoid, 1)
    pset.addPrimitive(custom_ops.tanh, 1)
    pset.addPrimitive(custom_ops.relu, 1)
    pset.addPrimitive(custom_ops.leaky_relu, 1)
    pset.addPrimitive(custom_ops.elu, 1)
    pset.addPrimitive(custom_ops.swish, 1)
    pset.renameArguments(ARG0='x', ARG1='wb1', ARG2='wb2')
    pset.activations = ['gelu', 'sigmoid', 'tanh', 'relu', 'leaky_relu', 'elu', 'swish']
    pset.all_linear_ids = [
        '1_1', '1_2', '1_3', '1_4',
        '2_1', '2_2', '2_3', '2_4',
        '3_1', '3_2', '3_3', '3_4',
        '4_1', '4_2', '4_3', '4_4',
    ]
    return pset
