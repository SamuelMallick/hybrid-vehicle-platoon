import pickle

import matplotlib.pyplot as plt
import numpy as np
from dmpcpwa.utils.tikz import save2tikz

plt.rc("text", usetex=True)
plt.rc("font", size=14)
plt.style.use("bmh")

nx_l = 2
sep = np.array([[-50], [0]])  # desired seperation between vehicles states
plot_len = 100
DG = False
Q = True
HOM = True
n = 10
N = 5
LT = 1

names = ["cent", "switching_admm", "seq" , "admm_100"]
error = []
cost = []

for name in names:
    with open(
        f"paper_2024_data/{name}_task_1_n_{n}_seed_2.pkl",
        "rb",
    ) as file:
        X = pickle.load(file)
        U = pickle.load(file)
        R = pickle.load(file)
        solve_times = pickle.load(file)
        node_counts = pickle.load(file)
        violations = pickle.load(file)
        leader_state = pickle.load(file)

        error.append(np.zeros((1, X.shape[0])))
        for t in range(X.shape[0]):
            error[-1][0, t] = np.linalg.norm(X[t, 0:nx_l] - leader_state[:, t])
            for i in range(1, n):
                error[-1][0, t] = np.linalg.norm(
                    X[[t], i * nx_l : (i + 1) * nx_l]
                    - X[[t], (i - 1) * nx_l : i * nx_l]
                    - sep.T
                )

        cost.append([sum(R[:t, 0, 0]) for t in range(R.shape[0])])

_, axs = plt.subplots(1, 1, constrained_layout=True, sharex=True)
for e in cost:
    axs.plot(e)

axs.set_xlabel(r"Time step $k$")
axs.set_ylabel(r"$\sum_0^k J(k)$")
axs.legend(["Centralized", "Sw-ADMM", "Sequential", "NC-ADMM"])
save2tikz(plt.gcf())
plt.show()
