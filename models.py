from torch import nn
from torchsummary import summary
from torchvision import models


class Flatten(nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


class FECNet(nn.Module):
    def __init__(self):
        super(FECNet, self).__init__()
        mobilenet = models.mobilenet_v2(pretrained=True)
        # Remove linear layer
        modules = list(mobilenet.children())[:-1]
        self.model = nn.Sequential(*modules,
                                   nn.AvgPool2d(kernel_size=7),
                                   Flatten(),
                                   nn.Linear(1280, 16),
                                   )
        # self.linear = nn.Linear(16, 1)
        # self.output = nn.Sigmoid()

    def forward(self, input):
        emb = self.model(input)
        return emb

    def predict(self, input):
        return self.model(input)


if __name__ == "__main__":
    from config import device

    model = FECNet().to(device)
    summary(model, input_size=[(3, 224, 224), (3, 224, 224)])
