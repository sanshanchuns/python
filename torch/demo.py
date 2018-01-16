import numpy as np
import torch

outs = []

# for i in range(10):
#     outs.append(torch.ones(1))
#
# print(outs)
# print(torch.stack(outs))
# print(torch.stack(outs, dim=1))

# a = np.arange(100).reshape(10, 10)
# print(a)
# b = np.random.choice(10, 3)
# print(a[b, :])

import gym
# import gym_pull
# import ppaquette_gym_super_mario
# gym_pull.pull('github.com/ppaquette/gym-super-mario')
# Only required once, envs will be loaded with import gym_pull afterwards
env = gym.make('SuperMarioBros-1-1-v0')
s = env.reset()
print(s.shape)
#
# while True:
#     env.render()
#     random_action = env.action_space.sample()
#     obser, re, done, info = env.step(random_action) # take a random action
#     if done:
#         print('done')
#         env.reset()

# import gym
# env = gym.make('CartPole-v0')
# env.reset()
# for _ in range(1000):
#     env.render()
#     random_action = env.action_space.sample()
#     obser, re, done, info = env.step(random_action) # take a random action
#     if done:
#         print('done')
#         print(obser)
#         # env.reset()

# N_ACTIONS = env.action_space.n
# N_STATES = env.observation_space.shape[0]

# for i_episode in range(20):
#     observation = env.reset()
#     for t in range(100):
#         env.render()
#         print(observation)
#         print(env.observation_space.shape[0])
#         print(env.action_space.n)
#         break
#         action = env.action_space.sample()
#         observation, reward, done, info = env.step(action)
#         if done:
#             print("Episode finished after {} timesteps".format(t+1))
#             break

x = torch.ones(1, 2)
y = torch.ones(1, 2)

print(np.random.uniform())

# y.add_(x)
# print(y)