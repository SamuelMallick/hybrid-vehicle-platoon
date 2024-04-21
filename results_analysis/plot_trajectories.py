import pickle

import matplotlib.pyplot as plt
import numpy as np

# from dmpcpwa.utils.tikz import save2tikz

plt.rc("text", usetex=True)
plt.rc("font", size=14)
plt.style.use("bmh")

nx_l = 2
plot_len = 100
name = "admm_50"
DG = False
Q = True
HOM = True
n = 5
N = 5
LT = 1
with open(
    f"data/task_2/{name}_n_task_2_{n}.pkl",
    "rb",
) as file:
    X = pickle.load(file)
    U = pickle.load(file)
    R = pickle.load(file)
    solve_times = pickle.load(file)
    node_counts = pickle.load(file)
    violations = pickle.load(file)
    leader_state = pickle.load(file)

print(f"tracking const: {sum(R)}")
print(f"av comp time: {sum(solve_times)/len(solve_times)}")

_, axs = plt.subplots(3, 1, constrained_layout=True, sharex=True)
axs[0].plot(leader_state[0, :plot_len], "--")
axs[1].plot(leader_state[1, :plot_len], "--")
for i in range(n):
    axs[0].plot(X[:plot_len, nx_l * i])
    axs[1].plot(X[:plot_len, nx_l * i + 1])
    if i > 0:
        axs[2].plot(X[:plot_len, nx_l * (i - 1)] - X[:plot_len, nx_l * (i)])
        axs[2].plot([0, plot_len], [25, 25], color="red")

ylim = 3000
axs[0].fill_between(
    np.linspace(0, 100, 100),
    ylim * np.ones(violations.shape),
    where=(violations > 0),
    color="red",
    alpha=0.3,
    label="Shaded Area",
)
axs[0].set_ylabel(r"pos ($m$)")
axs[1].set_ylabel(r"vel ($ms^{-1}$)")
# axs[1].set_ylim(0, 40)
# axs[0].set_ylim(0, ylim)
axs[2].set_xlabel(r"time step $k$")
axs[0].legend(["reference"])

# save2tikz(plt.gcf())

_, axs = plt.subplots(1, 1, constrained_layout=True, sharex=True)
axs.plot(U)

_, axs = plt.subplots(1, 1, constrained_layout=True, sharex=True)
axs.plot(R.squeeze())
plt.show()
