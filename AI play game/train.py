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
        # while True:
        for _ in range(100):
            # 可视化展示环境
            env.render()
            print("---输入当前状态---", s)
            action_value = dqn.choose_action(s) # action: tensor([0.0868, 0.1454], grad_fn=<AddBackward0>)
            print("---输出当前动作---", action_value)
            action = torch.argmax(action_value).numpy()
            y_eval = action_value[int(action)]
            next_s, reward, done, info = env.step(action)

            next_action_value = dqn.choose_action(next_s)
            dqn.learn(next_action_value, reward, y_eval)

