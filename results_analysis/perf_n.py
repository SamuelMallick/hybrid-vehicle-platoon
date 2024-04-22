import pickle

import matplotlib.pyplot as plt
import numpy as np

plt.rc("text", usetex=True)
plt.rc("font", size=14)
plt.style.use("bmh")

nx_l = 2

types = [
    "cent",
    "decent",
    "seq",
    "event_1",
    "event_5",
    "event_10",
    "admm_5",
    "admm_20",
    # "admm_50",
]
seeds = [0]
leg = [
    "decent",
    "seq",
    "event_1",
    "event_5",
    "event_10",
    "admm_5",
    "admm_20",
    # "admm_50",
]
num_event_vars = 3
num_admm_vars = 2

LT = 1
HOM = True
DG = False
Q = True
n_sw = [i for i in range(2, 3)]
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
        track_costs[counter].append(0)
        time_min[counter].append(0)
        time_max[counter].append(0)
        time_av[counter].append(0)
        nodes[counter].append(0)
        viols[counter].append(0)
        for seed in seeds:
            with open(
                f"data/{type}_task_1_n_{n}_seed_{seed}.pkl",
                "rb",
            ) as file:
                X = pickle.load(file)
                U = pickle.load(file)
                R = pickle.load(file)
                solve_times = pickle.load(file)
                node_counts = pickle.load(file)
                violations = pickle.load(file)
                leader_state = pickle.load(file)

            track_costs[counter][-1] += sum(R)
            time_min[counter][-1] += min(solve_times)[0]
            time_max[counter][-1] += max(solve_times)[0]
            time_av[counter][-1] += sum(solve_times)[0] / len(solve_times)
            nodes[counter][-1] += max(node_counts)
            viols[counter][-1] += sum(violations) / 100
        track_costs[counter][-1] /= 3
        time_min[counter][-1] /= 3
        time_max[counter][-1] /= 3
        time_av[counter][-1] /= 3
        nodes[counter][-1] /= 3
        viols[counter][-1] /= 3
    counter += 1


# plotting params for all figs
lw = 1.5  # line width
ms = 5  # marker size
mf = ["-x", "-o", "-o", ":v", ":v", ":v", "--s", "--s", "--s"]  # marker format

# tracking cost as percentrage performance drop from centralized
perf_drop = []
for i in range(1, counter):
    perf_drop.append(
        [
            100 * (track_costs[i][j] - track_costs[0][j]) / track_costs[i][j]
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

# plot perf drop
y_lim = 100
_, axs = plt.subplots(
    6,
    1,
    constrained_layout=True,
    sharex=True,
    gridspec_kw={"height_ratios": [1, 1, 0.6, 2, 1.2, 1]},
)
for i, leg_ in enumerate(leg):
    axs[0].plot(5, 5, mf[i + 1], label=leg_, markerfacecolor="none")
axs[0].set_axis_off()
axs[0].legend(leg, ncol=2, loc="center")
for i in range(counter - 1):
    axs[1].plot(
        n_sw,
        np.asarray(perf_drop[i]).reshape(len(n_sw)),
        mf[i + 1],
        linewidth=lw,
        markersize=ms,
        color=f"C{i}",
        markerfacecolor="none",
    )
axs[1].set_ylabel(r"$\%J$")
# axs[1].set_ylim(-25, y_lim)

y_lim = 0.3
for i in range(1, 3):
    _, _, bars = axs[2].errorbar(
        np.asarray([n_sw[j] + 0.0 * i for j in range(len(n_sw))]),
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
# axs[2].set_ylim(-0.1, y_lim)
axs[2].set_ylabel("$t_{COMP}$")
y_lim = 6
for i in range(3, 3 + num_event_vars):
    _, _, bars = axs[3].errorbar(
        np.asarray([n_sw[j] + 0.0 * i for j in range(len(n_sw))]),
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
# axs[3].set_ylim(-0.1, y_lim)
axs[3].set_ylabel("$t_{COMP}$")
y_lim = 3.5
for i in range(3 + num_event_vars, 3 + num_event_vars + num_admm_vars):
    _, _, bars = axs[4].errorbar(
        np.asarray([n_sw[j] + 0.0 * i for j in range(len(n_sw))]),
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
axs[4].set_ylabel("$t_{COMP}$")
# axs[4].set_ylim(-0.1, y_lim)
for i in range(1, counter):
    axs[5].plot(
        n_sw, nodes[i], mf[i], linewidth=lw, markersize=ms, markerfacecolor="none"
    )
axs[5].set_ylabel(r"$\#nodes$")
axs[5].set_yscale("log")
axs[5].set_xlabel("$n$")
# save2tikz(plt.gcf())

_, axs = plt.subplots(3, 1, constrained_layout=True, sharex=True)
for i in range(2):
    axs[0].plot(
        n_sw,
        np.asarray(perf_drop[i]).reshape(len(n_sw)),
        mf[i + 1],
        linewidth=lw,
        markersize=ms,
        color=f"C{i}",
        markerfacecolor="none",
    )
# axs[0].set_ylim(-25, y_lim)
for i in range(2, 2 + num_event_vars):
    axs[1].plot(
        n_sw,
        np.asarray(perf_drop[i]).reshape(len(n_sw)),
        mf[i + 1],
        linewidth=lw,
        markersize=ms,
        color=f"C{i}",
        markerfacecolor="none",
    )
# axs[1].set_ylim(-25, y_lim)
for i in range(2 + num_event_vars, 2 + num_event_vars + num_admm_vars):
    axs[2].plot(
        n_sw,
        np.asarray(perf_drop[i]).reshape(len(n_sw)),
        mf[i + 1],
        linewidth=lw,
        markersize=ms,
        color=f"C{i}",
        markerfacecolor="none",
    )
axs[2].set_xlabel("$n$")
# axs[2].set_ylim(-25, y_lim)
# save2tikz(plt.gcf())

# plot times
_, axs = plt.subplots(1, 1, constrained_layout=True, sharex=True)
y_lim = 5
for i in range(1, counter):
    _, _, bars = axs.errorbar(
        np.asarray([n_sw[j] + 0.0 * i for j in range(len(n_sw))]),
        np.asarray(time_av[i]),
        yerr=[np.asarray(error_lower[i]), np.asarray(error_upper[i])],
        linewidth=lw,
        markersize=ms,
        fmt=mf[i],
        capsize=4,
        markerfacecolor="none",
    )
    [bar.set_alpha(0.7) for bar in bars]
axs.legend(leg)
axs.set_xlabel("$n$")
axs.set_ylabel("$t_{COMP}$")
axs.set_ylim(-0.1, y_lim)
# save2tikz(plt.gcf())

_, axs = plt.subplots(3, 1, constrained_layout=True, sharex=True)
y_lim = 1000
for i in range(1, 3):
    axs[0].plot(n_sw, nodes[i], "-o", linewidth=lw, markersize=ms, color=f"C{i-1}")
axs[0].set_ylim(-0.1, y_lim)

for i in range(3, 3 + num_event_vars):
    axs[1].plot(n_sw, nodes[i], "-o", linewidth=lw, markersize=ms, color=f"C{i-1}")
axs[1].set_ylim(-0.1, y_lim)

for i in range(3 + num_event_vars, 3 + num_event_vars + num_admm_vars):
    axs[2].plot(n_sw, nodes[i], "-o", linewidth=lw, markersize=ms, color=f"C{i-1}")
axs[2].set_xlabel("$n$")
axs[2].set_ylim(-0.1, y_lim)
# save2tikz(plt.gcf())

_, axs = plt.subplots(1, 1, constrained_layout=True, sharex=True)
for i in range(1, counter):
    axs.plot(n_sw, viols[i], mf[i], linewidth=lw, markersize=ms, markerfacecolor="none")
axs.legend(leg)
axs.set_xlabel("$n$")
axs.set_ylabel(r"$\#CV$")
# save2tikz(plt.gcf())
plt.show()
