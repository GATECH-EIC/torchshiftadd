import unittest
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from huggingface_hub import hf_hub_download
from torchshiftadd import layers, models
from torchshiftadd.utils import test_acc

class ResNetShiftShapeTest(unittest.TestCase):

    def setup(self):
        self.input = torch.rand(2, 3, 32, 32)
        self.model = models.resnet20()
        models.convert_to_shift(self.model)

    def test_resnet_shift(self):
        self.setup()
        output = self.model(self.input)
        self.assertEqual(output.shape, (2, 10))

class ResNetShiftAccTest(unittest.TestCase):

    def setup(self):
        self.model = models.resnet20()
        models.convert_to_shift(self.model)
        self.model.cuda()
        self.test_dataloader = torch.utils.data.DataLoader(
            datasets.CIFAR10(
                './data.cifar10', 
                train=False, 
                download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ])
            ),
            batch_size=64, 
            shuffle=True,
        )

    def test_resnet_shift(self):
        self.setup()

        ckpt_path = hf_hub_download(
            repo_id="hryou1998/pretrained-ckpts",
            filename="resnet20-shift-cifar10.pth.tar",
        )

        checkpoint = torch.load(ckpt_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['state_dict'])
        models.convert_to_shift(self.model)

        top_1, top_5 = test_acc(self.model, self.test_dataloader)

        print("Top-1 Acc: {:.2f}%".format(top_1))
        print("Top-5 Acc: {:.2f}%".format(top_5))

if __name__ == "__main__":
    unittest.main()