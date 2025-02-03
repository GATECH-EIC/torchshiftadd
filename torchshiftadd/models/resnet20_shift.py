import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchshiftadd.layers import shift

def convert_to_shift(model):
    conversion_count = 0

    for name, module in reversed(model._modules.items()):

        if len(list(module.children())) > 0:
            model._modules[name], num_converted = convert_to_shift(model=module)
            conversion_count += num_converted
        
        if type(module) == nn.Conv2d:
            conv2d = module
            shift_conv2d = shift.Conv2dShift(
                module.in_channels, 
                module.out_channels, 
                module.kernel_size,
                module.stride,
                module.padding,
                module.dilation,
                module.groups,
                module.bias is not None,
                module.padding_mode
            )
            shift_conv2d.shift.data, shift_conv2d.sign.data = get_shift_and_sign(conv2d.weight)
            shift_conv2d.bias = conv2d.bias
            model._modules[name] = shift_conv2d
            conversion_count += 1

    return model, conversion_count

def get_shift_and_sign(x, rounding='deterministic'):
    sign = torch.sign(x)

    x_abs = torch.abs(x)
    shift = round(torch.log(x_abs) / np.log(2), rounding)

    return shift, sign

def round(x, rounding='deterministic'):
    assert(rounding in ['deterministic', 'stochastic'])
    if rounding == 'stochastic':
        x_floor = x.floor()
        return x_floor + torch.bernoulli(x - x_floor)
    else:
        return x.round()