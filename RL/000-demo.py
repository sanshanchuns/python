import pandas as pd
import numpy as np

ACTIONS = ['left', 'right']     # available actions

table = pd.DataFrame(
    np.zeros((6, 2)),
    columns=ACTIONS,
)

table.iloc[0, 1] = 1

actions = table.iloc[0, :]

# print(table.ix[0, 0])
# print(actions.idxmax())

print(range(2))
print(list(range(2)))

print(list(range(2, 4)))
