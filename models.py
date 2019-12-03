import torch
from torch import nn
from torchscope import scope
import torch.nn.functional as F
from mobilefacenet import MobileFaceNet


class FECNet(nn.Module):
    def __init__(self):
        super(FECNet, self).__init__()
        filename = 'mobilefacenet.pt'
        model = MobileFaceNet()
        model.load_state_dict(torch.load(filename))
        self.model = model
        self.fc = nn.Linear(128, 16)

    def forward(self, input):
        x = self.model(input)
        x = self.fc(x)
        x = F.normalize(x)
        return x


if __name__ == "__main__":
    model = FECNet()
    scope(model, input_size=(3, 112, 112))
