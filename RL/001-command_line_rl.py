import numpy as np
import pandas as pd
import time

LR = 0.1
EPSILON = 0.9
GAMMA = 0.9
EPISODES = 13
ACTIONS = ['left', 'right']
N_STATES = 6

np.random.seed(2)

def build_q_table(n_states, actions):
    return pd.DataFrame(
        np.zeros((n_states, len(actions))),
        columns=actions
    )

def choose_action(S, q_table):

    actions = q_table.iloc[S, :]

    if np.random.uniform() > EPSILON or actions.all() == 0:
        action = np.random.choice(ACTIONS)
    else:
        action = actions.idxmax()
    return action

def get_state_from_env(S, A):

    if A == 'right':
        if S == N_STATES - 2:
            S_ = 'terminal'
            R = 1
        else:
            S_ = S + 1
            R = 0
    else:
        R = 0
        if S == 0:
            S_ = S
        else:
            S_ = S - 1

    return S_, R

def update_env(S, episode, step_counter):
    env_list = ['-'] * (N_STATES -1) + ['T']

    if S == 'terminal':
        print('Episode: %d Steps: %d ' % (episode, step_counter))
    else:
        env_list[S] = 'o'
        print('\r{}'.format(''.join(env_list)), end='')

def lr():
    q_table = build_q_table(N_STATES, ACTIONS)

    for episode in range(EPISODES):
        S = 0
        step_counter = 0
        terminated = False

        update_env(S, episode, step_counter)

        while not terminated:

            A = choose_action(S, q_table)
            S_, R = get_state_from_env(S, A)
            q_predict = q_table.ix[S, A]

            if S_ == 'terminal':
                q_target = R
                terminated = True
            else:
                q_target = R + GAMMA * q_table.iloc[S_, :].max()

            q_table.ix[S, A] += LR * (q_target - q_predict)
            S = S_

            step_counter += 1
            update_env(S, episode, step_counter)

    print(q_table)

if __name__ == '__main__':
    lr()





