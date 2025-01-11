import os
import sys
import torch
import torch.nn as nn
import numpy as np

from torchshiftadd import utils

path = os.path.join(os.path.dirname(__file__), "extension")
adder_cuda = utils.load_extension(
    "adder_cuda", [
        os.path.join(path, "adder_cuda.cpp"),
        os.path.join(path, "adder_cuda_kernel.cu"),
    ]
)

class Adder2D(nn.Module):

    def __init__(self,
            input_channel,
            output_channel,
            kernel_size,
            stride = 1,
            padding = 0,
            groups = 1,
            bias = False,
            eta = 0.2,
        ):
        super(Adder2D, self).__init__()
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.kernel_size = kernel_size
        self.bias = bias
        self.eta = eta

        self.adder = torch.nn.Parameter(
            nn.init.normal_(
                torch.randn(output_channel, input_channel // groups, kernel_size, kernel_size)
            )
        )
        
        if self.bias:
            self.bias = torch.nn.Parameter(
                nn.init.uniform_(torch.zeros(output_channel))
            )
        else:
            self.bias = None

    def forward(self, input, ratio_out=1, ratio_in=1, ratio_g=1, kernel=None):

        sample_weight = self.adder[:(self.output_channel//ratio_out),:(self.input_channel//ratio_in),:,:]
        if (kernel!=None):
            start, end = sub_filter_start_end(5, kernel)
            sample_weight = sample_weight[:,:, start:end, start:end]
            padding = kernel//2
        else:
            padding = self.padding

        output = Adder2DFunction.apply(
            input,
            sample_weight,
            self.kernel_size,
            self.stride,
            padding,
            (self.groups//ratio_g),
            self.eta,
        )
        if self.bias is not None:
            output += self.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        return output


class Adder2DFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, kernel_size, stride, padding, groups, eta):
        ctx.save_for_backward(input, weight)
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.padding = padding
        ctx.groups = groups
        ctx.eta = eta
        ctx.quantize = False

        output = input.new_zeros(
            get_conv2d_output_shape(input, weight, stride, padding)
        )

        adder_cuda.forward(
            input,
            weight,
            output,
            kernel_size, kernel_size,
            stride, stride,
            padding, padding,
            groups, groups
        )

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        grad_input = grad_weight = None
        eta, kernel_size, stride, padding, groups = (
            ctx.eta, ctx.kernel_size, ctx.stride, ctx.padding, ctx.groups
        )

        # input
        if ctx.needs_input_grad[0]:
            grad_input = torch.zeros_like(input)
            adder_cuda.backward_input(
                grad_output,
                input,
                weight,
                grad_input,
                kernel_size, kernel_size,
                stride, stride,
                padding, padding,
                groups, groups
            )

        # weight
        if ctx.needs_input_grad[1]:
            grad_weight = torch.zeros_like(weight)
            adder_cuda.backward_weight(
                grad_output,
                input,
                weight,
                grad_weight,
                kernel_size, kernel_size,
                stride, stride,
                padding, padding,
                groups, groups)
            grad_weight = eta * np.sqrt(grad_weight.numel()) / torch.norm(grad_weight).clamp(min=1e-12) * grad_weight

        return grad_input, grad_weight, None, None, None, None, None


def get_conv2d_output_shape(input, weight, stride, padding):
    n_filters, d_filter, h_filter, w_filter = weight.size()
    n_x, d_x, h_x, w_x = input.size()

    h_out = (h_x - h_filter + 2 * padding) // stride + 1
    w_out = (w_x - w_filter + 2 * padding) // stride + 1

    return (n_x, n_filters, h_out, w_out)


def sub_filter_start_end(kernel_size, sub_kernel_size):
    center = kernel_size // 2
    dev = sub_kernel_size // 2
    start, end = center - dev, center + dev + 1
    assert end - start == sub_kernel_size
    return start, end