import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from dmpcpwa.utils.tikz import save2tikz

plt.style.use("bmh")
plt.rc("text", usetex=True)
plt.rc("font", size=14)

fig, axs = plt.subplots(5, 2, constrained_layout=True, sharex=True, sharey=True)

names = ["cent", "decent", "seq", "event6", "admm20"]

n_range = [i for i in range(2, 11)]
N_range = [i for i in range(2, 6)]

data = [np.zeros((len(n_range), len(N_range))) for i in range(10)]
i = 0
j = 0
counter = 0
for name in names:
    col = 0
    for Q in [False, True]:
        i = 0
        for n in n_range:
            j = 0
            for N in N_range:
                with open(
                    f"{name}_n_{n}_N_{N}_Q_{Q}_DG_False_HOM_True_LT_1.pkl",
                    "rb",
                ) as file:
                    X = pickle.load(file)
                    U = pickle.load(file)
                    R = pickle.load(file)
                    solve_times = pickle.load(file)
                    node_counts = pickle.load(file)
                    violations = pickle.load(file)
                    leader_state = pickle.load(file)
                data[counter][i, j] = sum(solve_times) / len(solve_times)
                j += 1
            i += 1
        counter += 1

min_vals = min([np.min(data[i]) for i in range(10)])
max_vals = max([np.max(data[i]) for i in range(10)])

counter = 0
for row in range(5):
    # Create a heatmap using seaborn
    min_val = min(np.min(data[counter]), np.min(data[counter + 1]))
    max_val = max(np.max(data[counter]), np.max(data[counter + 1]))
    sns.heatmap(
        data[counter],
        vmin=min_val,
        vmax=max_val,
        annot=False,
        cbar=False,
        cmap="Reds",
        fmt=".2f",
        ax=axs[row, 0],
    )
    sns.heatmap(
        data[counter + 1],
        vmin=min_val,
        vmax=max_val,
        annot=False,
        cbar=True,
        cmap="Reds",
        fmt=".2f",
        ax=axs[row, 1],
    )
    axs[row, 0].set_yticks([0.5, 8.5])
    axs[row, 0].set_yticklabels(["2", "10"])
    counter += 2

axs[4, 0].set_xticks([0.5, 3.5])
axs[4, 1].set_xticklabels(["2", "5"])

axs[0, 0].set_ylabel(r"$n$ - cent")
axs[1, 0].set_ylabel(r"$n$ - decent")
axs[2, 0].set_ylabel(r"$n$ - seq")
axs[3, 0].set_ylabel(r"$n$ - event (4)")
axs[4, 0].set_ylabel(r"$n$ - admm (20)")

axs[4, 0].set_xlabel(r"$N - \|\cdot\|_1$")
axs[4, 1].set_xlabel(r"$N - \|\cdot\|_2^2$")

save2tikz(plt.gcf())

# Add labels to axes
# fig.text(0.5, 0, 'N', ha='center')
# fig.text(0, 0.5, 'n', va='center', rotation='vertical')

# Show the plot
plt.show()
