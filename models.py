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
    def __init__(self, pretrained=True):
        super(RankNetMobile, self).__init__()
        # mobilenet = models.mobilenet_v2(pretrained=True)

        filename = 'mobilefacenet.pt'
        model = MobileFaceNet()
        if pretrained:
            model.load_state_dict(torch.load(filename))

        self.model = model
        self.dropout = nn.Dropout(0.8)
        # self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.fc = nn.Linear(128, 16)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input1, input2, input3):
        e1 = self.predict(input1)
        e2 = self.predict(input2)
        e3 = self.predict(input3)
        d12 = F.pairwise_distance(e1, e2, p=2)
        d13 = F.pairwise_distance(e1, e3, p=2)
        # d23 = F.pairwise_distance(e2, e3, p=2)
        return self.sigmoid(d12 - d13)

    def predict(self, input):
        x = self.model(input)
        x = self.dropout(x)
        # x = self.relu(x)
        x = self.fc(x)
        x = F.normalize(x, dim=1)
        return x


class ResNetEmotionModel(nn.Module):
    def __init__(self):
        super(ResNetEmotionModel, self).__init__()
        resnet = models.resnet50(pretrained=True)
        # Remove linear layer
        modules = list(resnet.children())[:-1]
        self.features = nn.Sequential(*modules)
        # building last several layers
        self.fc = nn.Linear(2048, 16)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.normalize(x)
        return x


class ResNetRankModel(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNetRankModel, self).__init__()
        resnet = models.resnet50(pretrained=True)
        # Remove linear layer
        modules = list(resnet.children())[:-1]
        self.features = nn.Sequential(*modules)
        self.fc = nn.Linear(2048, 16)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input1, input2, input3):
        e1 = self.predict(input1)
        e2 = self.predict(input2)
        e3 = self.predict(input3)
        d12 = F.pairwise_distance(e1, e2, p=2)
        d13 = F.pairwise_distance(e1, e3, p=2)
        d23 = F.pairwise_distance(e2, e3, p=2)

        return self.sigmoid(d12 - (d13 + d23) / 2)

    def predict(self, input):
        x = self.features(input)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.normalize(x)
        return x


if __name__ == "__main__":
    model = ResNetEmotionModel()
    scope(model, input_size=(3, 224, 224))
