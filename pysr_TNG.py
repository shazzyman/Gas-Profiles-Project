import pysr
pysr.install(precompile=False)  # Ensures PySR installs dependencies

import sympy
import numpy as np
from matplotlib import pyplot as plt
from pysr import PySRRegressor
from sklearn.model_selection import train_test_split
import Vik_density_compare as Vik  

# **Setup Instructions:**
# 1. Install Julia: Ensure Julia is installed and accessible from your terminal.
# 2. Install PySR: Install using `pip install pysr`.
# 3. Ensure any required Python libraries (numpy, matplotlib, sklearn) are installed.

# power_operator = {
#     "pow": (lambda x, y: np.power(x, y), 2),
# }

X = Vik.my_bins100_centers
X = X.reshape(-1, 1)

r500 = []
for i in range(len(X)):
    if X[i] > 1:
        r500 = np.append(r500, i)
        break
        
X1 = X[1:int(r500[0])]     
y1 = (Vik.median_rho_100[1:int(r500[0])])

X2 = X[int(r500[0]):32]
y2 = (Vik.median_rho_100[int(r500[0]):32])

X3 = X[32:-1]
y3 =(Vik.median_rho_100[32:-1])



default_pysr_params = dict(
    populations=50,
    maxsize = 20,
    model_selection="best",
)


model = PySRRegressor(
    niterations=500,
    binary_operators=["+", "-", "*", "/", "^"],
    unary_operators= ["exp", "log10", "log"],
    complexity_of_operators= {"exp": 2},
    **default_pysr_params
)

model.fit(X1, y1)
model.fit(X2, y2)
model.fit(X3, y3)


