import pandas as pd
import numpy as np

ACTIONS = ['left', 'right']     # available actions

table = pd.DataFrame(
    np.zeros((6, 2)),
    columns=ACTIONS,
)

# table.iloc[0, 0] = 1

print(type(table))
print(table)

state_actions = table.iloc[0, :]
print(type(state_actions))

print(state_actions.idxmax())
if not state_actions.any():
    print(state_actions.all())

if state_actions.all() == 0:
    print('all zero')