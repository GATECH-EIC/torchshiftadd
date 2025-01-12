import unittest
import torch

from torchshiftadd import layers

class LinearShiftTest(unittest.TestCase):

    def setup(self):
        self.input = torch.rand(32, 32)

    def test_adder2d(self):
        self.setup()
        shift = layers.LinearShift(
            in_features=32,
            out_features=64,
            bias=True,
        )
        output = shift(self.input)
        self.assertEqual(output.shape, (32, 64))

class Conv2dShiftTest(unittest.TestCase):

    def setup(self):
        self.input = torch.rand(1, 3, 32, 32)
        self.weight = torch.rand(64, 3, 3, 3)
        self.bias = torch.rand(64)
        self.stride = 1
        self.padding = 1
        self.groups = 1

    def test_adder2d(self):
        self.setup()
        shift = layers.Conv2dShift(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=self.stride,
            padding=self.padding,
            groups=self.groups,
            bias=True,
            weight_bits=4,
            input_bits=16,
        )
        output = shift(self.input)
        self.assertEqual(output.shape, (1, 64, 32, 32))

if __name__ == "__main__":
    unittest.main()