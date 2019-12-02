from torch import nn
from torch.nn import functional as F
from torchscope import scope
from torchvision import models
from mobilefacenet import MobileFaceNet
import torch

class Flatten(nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


class FECNet(nn.Module):
    def __init__(self):
        super(FECNet, self).__init__()
        filename = 'mobilefacenet.pt'
        model = MobileFaceNet()
        model.load_state_dict(torch.load(filename))

        # Remove linear layer
        modules = list(model.children())
        self.model = nn.Sequential(*modules,
                                   # nn.AvgPool2d(kernel_size=7),
                                   # Flatten(),
                                   # nn.Linear(1280, 16),
                                   )

    def forward(self, input):
        x = self.model(input)
        # x = F.normalize(x)
        return x


if __name__ == "__main__":
    from config import device

    model = FECNet().to(device)
    scope(model, input_size=[(3, 112, 112), (3, 112, 112)])
