import torch
import torch.nn.functional as F
from torch import nn
from torchscope import scope
from torchvision import models

from mobilefacenet import MobileFaceNet


class FECNet(nn.Module):
    def __init__(self):
        super(FECNet, self).__init__()
        filename = 'mobilefacenet.pt'
        model = MobileFaceNet()
        model.load_state_dict(torch.load(filename))
        self.model = model
        self.relu = nn.PReLU()
        self.fc = nn.Linear(128, 16)

    def forward(self, input):
        x = self.model(input)
        x = self.relu(x)
        x = self.fc(x)
        x = F.normalize(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, nin, nout, kernel_size, padding, bias=False):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size, padding=padding, groups=nin, bias=bias)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class RankNetMobile(nn.Module):
    def __init__(self):
        super(RankNetMobile, self).__init__()
        # mobilenet = models.mobilenet_v2(pretrained=True)
        filename = 'mobilefacenet.pt'
        model = MobileFaceNet()
        model.load_state_dict(torch.load(filename))
        # Remove linear layer
        modules = list(model.children())
        self.model = nn.Sequential(*modules,
                                   # nn.AvgPool2d(kernel_size=7),
                                   # DepthwiseSeparableConv(1280, 1280, kernel_size=4, padding=0),
                                   # Flatten(),
                                   nn.Dropout(0.5),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Linear(128, 16),
                                   # nn.Sigmoid(),
                                   )
        self.output = nn.Sigmoid()

    def forward(self, input1, input2, input3):
        e1 = self.model(input1)
        e2 = self.model(input2)
        e3 = self.model(input3)
        s1 = F.pairwise_distance(e1, e2, p=2)
        s2 = F.pairwise_distance(e2, e3, p=2)
        prob = self.output(s1 - s2)
        return prob

    def predict(self, input):
        s = self.model(input)
        return self.output(s)


if __name__ == "__main__":
    model = FECNet()
    scope(model, input_size=(3, 112, 112))
