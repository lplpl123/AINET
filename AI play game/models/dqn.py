import torch

from models import net
from torch import optim
from torch import nn


class DQN:

    def __init__(self):
        # 定义参数
        self.eval_net = net.Net()
        self.target_net = net.Net()
        self.optimizer = optim.Adam(self.eval_net.parameters())
        self.loss_func = nn.MSELoss()

    def choose_action(self, x): # x: [ 0.02234908 -0.04540155  0.03452853  0.00153637]
        x = torch.tensor(x) # x: tensor([ 0.0223, -0.0454,  0.0345,  0.0015])
        x = self.eval_net.forward(x)
        # todo
        return x

    def learn(self, next_action_value, reward, y_eval):
        y_target = reward + 0.9 * max(next_action_value)
        loss = self.loss_func(y_eval, y_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
