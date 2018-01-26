
from maze_env import Maze
from RL_brain import QLearningTable

def update():

    for episode in range(100):

        state = env.reset()

        while True:

            action = RL.choose_action(str(state))
            state_, reward, done = env.step(action)
            RL.learn(str(state), reward, action, str(state_))
            state = state_

            if done:
                break

    print('game over')
    env.destroy()


if __name__ == '__main__':
    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()
