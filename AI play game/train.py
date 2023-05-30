import gym
import torch
from models import dqn


if __name__ == "__main__":
    # ini
    EPOCHES = 2
    env = gym.make('CartPole-v0')
    dqn = dqn.DQN()

    for epoch in range(EPOCHES):
        s = env.reset()
        while True:
            # 可视化展示环境
            env.render()
            action_value = dqn.choose_action(s) # action: tensor([0.0868, 0.1454], grad_fn=<AddBackward0>)
            action = torch.argmax(action_value).numpy()
            next_s, reward, done, info = env.step(action)

            x, x_dot, theta, theta_dat = next_s

            break
