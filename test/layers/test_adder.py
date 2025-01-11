import unittest
import torch

import torchshiftadd
from torchshiftadd import layers

class Adder2DTest(unittest.TestCase):

    def setup(self):
        self.input = torch.rand(1, 3, 32, 32)
        self.weight = torch.rand(64, 3, 3, 3)
        self.bias = torch.rand(64)
        self.stride = 1
        self.padding = 1
        self.groups = 1
        self.eta = 1.0

    def test_adder2d(self):
        self.setup()
        adder = layers.Adder2D(
            input_channel=3,
            output_channel=64,
            kernel_size=3,
            stride=self.stride,
            padding=self.padding,
            groups=self.groups,
            bias=True,
            eta=self.eta,
        )
        output = adder(self.input)
        self.assertEqual(output.shape, (1, 64, 32, 32))

if __name__ == "__main__":
    unittest.main()