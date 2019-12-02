import torch
from torch import nn
from torchscope import scope

from mobilefacenet import MobileFaceNet


class FECNet(nn.Module):
    def __init__(self):
        super(FECNet, self).__init__()
        filename = 'mobilefacenet.pt'
        self.model = MobileFaceNet()
        model.load_state_dict(torch.load(filename))

        # # Remove linear layer
        # modules = list(model.children())
        # self.model = nn.Sequential(*modules,
        #                            # nn.Linear(1280, 16),
        #                            )

    def forward(self, input):
        x = self.model(input)
        # x = F.normalize(x)
        return x


if __name__ == "__main__":
    model = FECNet()
    scope(model, input_size=(3, 112, 112))
