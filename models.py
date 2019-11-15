import torch
from torch import nn
from torchsummary import summary
from torchvision import models


class Flatten(nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


class RankNetMobile(nn.Module):
    def __init__(self):
        super(RankNetMobile, self).__init__()
        mobilenet = models.mobilenet_v2(pretrained=True)
        # Remove linear layer
        modules = list(mobilenet.children())[:-1]
        self.model = nn.Sequential(*modules,
                                   nn.AvgPool2d(kernel_size=7),
                                   Flatten(),
                                   nn.Linear(1280, 16),
                                   )
        self.linear = nn.Linear(16, 1)
        self.output = nn.Sigmoid()

    def forward(self, input1, input2, input3):
        emb1 = self.linear(self.model(input1))
        print('emb1.size(): ' + str(emb1.size()))
        emb2 = self.linear(self.model(input2))
        emb3 = self.linear(self.model(input3))
        s1 = torch.dist(emb1, emb2)
        print('s1.size(): ' + str(s1.size()))
        s2 = torch.dist(emb1, emb3)
        s3 = torch.dist(emb2, emb3)
        prob = (self.output(s1 - s2) + self.output(s1 - s3)) / 2
        print('prob.size(): ' + str(prob.size()))
        return prob

    def predict(self, input):
        return self.model(input)


if __name__ == "__main__":
    from config import device

    model = RankNetMobile().to(device)
    summary(model, input_size=[(3, 224, 224), (3, 224, 224)])
