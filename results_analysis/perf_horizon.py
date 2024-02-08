import pickle

import matplotlib.pyplot as plt
import numpy as np
from dmpcpwa.utils.tikz import save2tikz

plt.rc("text", usetex=True)
plt.rc("font", size=14)
plt.style.use("bmh")

nx_l = 2

types = [
    "cent",
    "decent",
    "seq",
    "event4",
    "event6",
    "event10",
    "admm20",
    "admm50",
]
leg = [
    "decent",
    "seq",
    "event (4)",
    "event (6)",
    "event (10)",
    "admm (20)",
    "admm (50)",
]
num_event_vars = 3
num_admm_vars = 2

LT = 1
HOM = True
DG = False
Q = True
N_sw = [i for i in range(2, 9)]
n = 10

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
    for N in N_sw:
        with open(
            f"{type}_n_{n}_N_{N}_Q_{Q}_DG_{DG}_HOM_{HOM}_LT_{LT}.pkl",
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
        time_min[counter].append(min(solve_times)[0])
        time_max[counter].append(max(solve_times)[0])
        time_av[counter].append(sum(solve_times)[0] / len(solve_times))
        nodes[counter].append(max(node_counts))
        viols[counter].append(sum(violations) / 100)
    counter += 1


lw = 1.5
ms = 5
mf = ["-x", "-o", "-o", ":v", ":v", ":v", "--s", "--s"]  # marker format

# plot centralized tracking cost with respect to N
plt.plot(np.asarray(track_costs[0]).squeeze())

# tracking cost
perf_drop = []
for i in range(1, counter):
    perf_drop.append(
        [
            track_costs[i][j][0, 0] - track_costs[0][j][0, 0]
            for j in range(len(track_costs[0]))
        ]
    )

# calculate time error bars
_, axs = plt.subplots(1, 1, constrained_layout=True, sharex=True)
error_lower = [
    [time_av[i][j] - time_min[i][j] for j in range(len(N_sw))] for i in range(counter)
]
error_upper = [
    [time_max[i][j] - time_av[i][j] for j in range(len(N_sw))] for i in range(counter)
]

_, axs = plt.subplots(5, 1, constrained_layout=True, sharex=True)

# create empty lefend plot
for i, leg_ in enumerate(leg):
    axs[0].plot(5, 5, mf[i + 1], label=leg_, markerfacecolor="none")
axs[0].set_axis_off()
axs[0].legend(leg, ncol=3, loc="center")

for i in range(counter - 1):
    axs[1].plot(
        N_sw,
        np.asarray(perf_drop[i]).reshape(len(N_sw)),
        mf[i + 1],
        linewidth=lw,
        markersize=ms,
        color=f"C{i}",
        markerfacecolor="none",
    )
axs[1].set_ylabel(r"$J$")
# axs[1].set_yscale('log')

y_lim = 20
for i in range(1, counter):
    _, _, bars = axs[2].errorbar(
        np.asarray([N_sw[j] + 0.0 * i for j in range(len(N_sw))]),
        np.asarray(time_av[i]),
        yerr=[np.asarray(error_lower[i]), np.asarray(error_upper[i])],
        linewidth=lw,
        markersize=ms,
        color=f"C{i-1}",
        fmt=mf[i],
        capsize=4,
        markerfacecolor="none",
    )
    [bar.set_alpha(0.7) for bar in bars]
axs[2].set_ylim(-0.1, y_lim)
# axs[2].set_yscale('log')
for i in range(1, counter):
    axs[3].plot(
        N_sw, nodes[i], mf[i], linewidth=lw, markersize=ms, markerfacecolor="none"
    )
axs[3].set_ylabel(r"$\#nodes$")
# axs[3].set_yscale('log')


for i in range(1, counter):
    axs[4].plot(
        N_sw, viols[i], mf[i], linewidth=lw, markersize=ms, markerfacecolor="none"
    )
axs[4].set_xlabel("$N$")
axs[4].set_ylabel(r"$\#CV$")
save2tikz(plt.gcf())
plt.show()
