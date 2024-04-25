import pickle

import matplotlib.pyplot as plt
from dmpcpwa.utils.tikz import save2tikz

plt.rc("text", usetex=True)
plt.rc("font", size=14)
plt.style.use("bmh")

nx_l = 2

types = ["cent", "switching_admm", "seq", "admm_100"]
leg = ["centralized", "Sw-ADMM", "sequential", "NC-ADMM"]

LT = 1
HOM = True
DG = False
Q = True
n_sw = [5, 10, 15]
N = 5

track_costs = []
time_min = []
time_max = []
time_av = []
nodes = []
viols = []
counter = 0
for type in types:
    track_costs.append([])
    time_min.append([])
    time_max.append([])
    time_av.append([])
    nodes.append([])
    viols.append([])
    for n in n_sw:
        with open(
            f"paper_2024_data/{type}_task_1_n_{n}_seed_2.pkl",
            "rb",
        ) as file:
            X = pickle.load(file)
            U = pickle.load(file)
            R = pickle.load(file)
            solve_times = pickle.load(file)
            node_counts = pickle.load(file)
            violations = pickle.load(file)
            leader_state = pickle.load(file)

        track_costs[counter].append(sum(R))
        if isinstance(solve_times, list):
            time_min[counter].append(min(solve_times))
            time_max[counter].append(max(solve_times))
            time_av[counter].append(sum(solve_times) / len(solve_times))
        else:
            time_min[counter].append(min(solve_times)[0])
            time_max[counter].append(max(solve_times)[0])
            time_av[counter].append(sum(solve_times)[0] / len(solve_times))

    counter += 1

# tracking cost as percentrage performance drop from centralized
perf_drop = []
for i in range(1, counter):
    perf_drop.append(
        [
            100 * (track_costs[i][j] - track_costs[0][j]) / track_costs[0][j]
            for j in range(len(track_costs[0]))
        ]
    )
# calculate time error bars
error_lower = [
    [time_av[i][j] - time_min[i][j] for j in range(len(n_sw))] for i in range(counter)
]
error_upper = [
    [time_max[i][j] - time_av[i][j] for j in range(len(n_sw))] for i in range(counter)
]

lw = 1
ms = 5
markers = ["o", "*", "s", "D"]
_, axs = plt.subplots(2, 1, constrained_layout=True, sharex=True)
for i in range(len(types)):
    axs[0].plot(
        n_sw,
        [track_costs[i][j][0, 0] for j in range(len(track_costs[i]))],
        "--o",
        color=f"C{i}",
        linewidth=lw,
        markersize=ms,
    )
    axs[1].plot(
        n_sw,
        time_av[i],
        marker="o",
        linestyle="--",
        color=f"C{i}",
        linewidth=lw,
        label="_nolegend_",
        markersize=ms,
    )
    axs[1].plot(
        n_sw,
        time_min[i],
        marker="^",
        linestyle="None",
        color=f"C{i}",
        label="_nolegend_",
        markersize=ms,
    )
    axs[1].plot(
        n_sw,
        time_max[i],
        marker="v",
        linestyle="None",
        color=f"C{i}",
        label="_nolegend_",
        markersize=ms,
    )

    for j in range(len(n_sw)):
        axs[1].plot(
            [n_sw[j], n_sw[j]],
            [time_max[i][j], time_min[i][j]],
            color=f"C{i}",
            linewidth=lw,
            label="_nolegend_",
            markersize=ms,
        )

axs[1].set_yscale("log")
axs[0].legend(leg, ncol=2)
axs[0].set_ylabel(r"$\sum_{t=0}^T J(t)$")
axs[1].set_ylabel(r"Time (s)")
axs[1].set_xlabel(r"$M$")
save2tikz(plt.gcf())
plt.show()
