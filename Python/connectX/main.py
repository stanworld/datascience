import numpy as np
from kaggle_environments import make
env = make("connectx", debug=True)
print(list(env.agents))
