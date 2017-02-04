import gym

env = gym.make("SpaceInvaders-v0")

env.reset()

#for _ in range(1000):
for i in range(2000):
    env.render()
    env.step(env.action_space.sample())
    if (i % 100 == 0):
        print(i)