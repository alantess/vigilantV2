import os
import torch
import torchvision as vision
from torch import nn


class LanesSegNet(nn.Module):
    def __init__(self, num_classes=3, chkpt='saved_models'):
        super(LanesSegNet, self).__init__()
        self.chkpt = chkpt
        self.file = os.path.join(chkpt, "lanes_segnet.pt")
        self.base = vision.models.segmentation.deeplabv3_resnet101(
            False, num_classes=num_classes)

    def forward(self, x):
        return self.base(x)['out']

    def save(self):
        if not os.path.exists(self.chkpt):
            os.mkdir(self.chkpt)
        torch.save(self.state_dict(), self.file)

    def load(self):
        self.load_state_dict(torch.load(self.file))
