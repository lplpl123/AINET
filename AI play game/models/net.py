from torch import nn
from torch.nn import functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, 50)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(50, 2)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.out(x)
        return x