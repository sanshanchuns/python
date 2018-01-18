import pandas as pd
import numpy as np

ACTIONS = ['left', 'right']     # available actions

table = pd.DataFrame(
    np.ones((6, 2)),
    columns=ACTIONS,
)

table.iloc[0, 1] = 0

print(type(table))
print(table)

state_actions = table.iloc[0, :]
print(type(state_actions))
if state_actions.all() == 0:
    print(state_actions.all())