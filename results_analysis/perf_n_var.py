import pickle

from matplotlib import ticker
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statistics
from matplotlib.gridspec import GridSpec

plt.rc("text", usetex=True)
plt.rc("font", size=14)
plt.style.use("bmh")

nx_l = 2

q_du_pen = 0
harmonic_mean = False
stand_error = False
multi_lead = True

types = [
    "cent",
    "decent_vest_False",
    # "decent_vest_True",
    # "decent_vest_sat",
    # "seq",
    # "event_1",
    # "event_5",
    # "event_10",
    # "admm_5",
    "admm_20",
    # "admm_50",
]
leg = [
    "decent",
    # "decent_pred_1",
    # "decent_pred_2",
    # "seq",
    # "event_1",
    # "event_5",
    # "event_10",
    # "admm_5",
    "admm_20",
    # "admm_50",
]
seeds = [i for i in range(1) if i not in [3]]
num_decent_vars = 1
num_seq_vars = 0
num_event_vars = 0
num_admm_vars = 1

n_sw = [i for i in range(2, 9)]
N = 6

track_costs = []
time_min = []
time_max = []
time_av = []
time = []
nodes = []
viols = []

for type_indx, type in enumerate(types):
    track_costs.append([])
    time_min.append([])
    time_max.append([])
    time_av.append([])
    time.append([])
    nodes.append([])
    viols.append([])
    for n_indx, n in enumerate(n_sw):
        track_costs[type_indx].append([])
        time_min[type_indx].append([])
        time_max[type_indx].append([])
        time_av[type_indx].append([])
        time[type_indx].append([])
        nodes[type_indx].append([])
        viols[type_indx].append([])
        for seed_indx, seed in enumerate(seeds):
            track_costs[type_indx][n_indx].append([0])
            time_min[type_indx][n_indx].append([0])
            time_max[type_indx][n_indx].append([0])
            time_av[type_indx][n_indx].append([0])
            viols[type_indx][n_indx].append([0])
            if multi_lead:
                leads = [i for i in range(0, n)]
            else:
                leads = [0]

            for lead in leads:
                if not multi_lead:
                    file_name = (
                        f"data/{type}_task_2_n_{n}_N_{N}_q_0.2_seed_{seed}.pkl"
                        if q_du_pen != 0
                        else f"data/{type}_task_2_n_{n}_N_{N}_seed_{seed}.pkl"
                    )
                else:
                    file_name = f"data/multi_leader/{type}_task_2_n_{n}_N_{N}_lead_{lead}_seed_{seed}.pkl"
                try:
                    with open(
                        file_name,
                        "rb",
                    ) as file:
                        X = pickle.load(file)
                        U = pickle.load(file)
                        R = pickle.load(file)
                        solve_times = pickle.load(file)
                        node_counts = pickle.load(file)
                        violations = pickle.load(file)
                        leader_state = pickle.load(file)

                    track_costs[type_indx][n_indx][seed_indx] += sum(R)[0, 0]
                    time_min[type_indx][n_indx][seed_indx] += min(solve_times)[0]
                    time_max[type_indx][n_indx][seed_indx] += max(solve_times)[0]
                    time_av[type_indx][n_indx][seed_indx] += sum(solve_times)[0] / len(
                        solve_times
                    )
                    time[type_indx][n_indx] += list(solve_times.squeeze())
                    nodes[type_indx][n_indx] += list(node_counts.squeeze())
                    v = sum(violations) / 100
                    if v != 0:
                        pass
                    viols[type_indx][n_indx][seed_indx] += v
                except:
                    print(f"no seed {seed}")
        pass


# plotting params for all figs
lw = 1.5  # line width
ms = 5  # marker size
mf = (
    ["-o"] * num_decent_vars
    + ["-x"] * num_seq_vars
    + [":v"] * num_event_vars
    + ["--s"] * num_admm_vars
)  # marker format

# tracking cost as percentrage performance drop from centralized
perf_drop_abs = [None] * (len(types) - 1)
perf_drop_rel = [None] * (len(types) - 1)
for i in range(1, len(types)):
    perf_drop_rel[i - 1] = [
        [
            100
            * (track_costs[i][j][k][0] - track_costs[0][j][k][0])
            / track_costs[0][j][k][0]
            for k in range(len(seeds))
        ]
        for j in range(len(n_sw))
    ]
    perf_drop_abs[i - 1] = [
        [(track_costs[i][j][k][0] - track_costs[0][j][k][0]) for k in range(len(seeds))]
        for j in range(len(n_sw))
    ]

# get rid of the dimension
viols = [
    [[x[0] for x in viols[i][j]] for j in range(len(n_sw))] for i in range(len(types))
]

if stand_error:
    perf_drop_rel_sd = [
        [
            np.std(perf_drop_rel[i][j]) / np.sqrt(len(perf_drop_rel[i][j]))
            for j in range(len(n_sw))
        ]
        for i in range(len(perf_drop_rel))
    ]
    perf_drop_abs_sd = [
        [
            np.std(perf_drop_abs[i][j]) / np.sqrt(len(perf_drop_abs[i][j]))
            for j in range(len(n_sw))
        ]
        for i in range(len(perf_drop_abs))
    ]

    viols_sd = [
        [np.std(viols[i][j]) / np.sqrt(len(viols[i][j])) for j in range(len(n_sw))]
        for i in range(1, len(viols))
    ]
else:
    perf_drop_rel_sd = [
        [np.std(perf_drop_rel[i][j]) for j in range(len(n_sw))]
        for i in range(len(perf_drop_rel))
    ]
    perf_drop_abs_sd = [
        [np.std(perf_drop_abs[i][j]) for j in range(len(n_sw))]
        for i in range(len(perf_drop_abs))
    ]

    viols_sd = [
        [np.std(viols[i][j]) for j in range(len(n_sw))] for i in range(1, len(viols))
    ]
if harmonic_mean:
    # add negative values so that harmonic mean has only positives. Then subtract the min from the mean
    min_rel = min(
        min(
            min(
                [
                    [min(perf_drop_rel[i][j]) for j in range(len(n_sw))]
                    for i in range(1, len(perf_drop_rel))
                ]
            )
        ),
        0,
    )
    min_abs = min(
        min(
            min(
                [
                    [min(perf_drop_abs[i][j]) for j in range(len(n_sw))]
                    for i in range(1, len(perf_drop_abs))
                ]
            )
        ),
        0,
    )
    perf_drop_rel_mean = [
        [
            statistics.harmonic_mean([item - min_rel for item in perf_drop_rel[i][j]])
            for j in range(len(n_sw))
        ]
        for i in range(len(perf_drop_rel))
    ]
    perf_drop_abs_mean = [
        [
            statistics.harmonic_mean([item - min_abs for item in perf_drop_abs[i][j]])
            for j in range(len(n_sw))
        ]
        for i in range(len(perf_drop_abs))
    ]

    perf_drop_rel_mean = [
        [item - min_rel for item in perf_drop_rel_mean[i]]
        for i in range(len(perf_drop_rel_mean))
    ]
    perf_drop_abs_mean = [
        [item - min_rel for item in perf_drop_abs_mean[i]]
        for i in range(len(perf_drop_abs_mean))
    ]

    time_mean = [
        [statistics.harmonic_mean(time[i][j]) for j in range(len(n_sw))]
        for i in range(1, len(time))
    ]

    viols_mean = [
        [statistics.harmonic_mean(viols[i][j]) for j in range(len(n_sw))]
        for i in range(1, len(viols))
    ]
else:
    perf_drop_rel_mean = [
        [np.mean(perf_drop_rel[i][j]) for j in range(len(n_sw))]
        for i in range(len(perf_drop_rel))
    ]
    perf_drop_abs_mean = [
        [np.mean(perf_drop_abs[i][j]) for j in range(len(n_sw))]
        for i in range(len(perf_drop_abs))
    ]

    time_mean = [
        [np.mean(time[i][j]) for j in range(len(n_sw))] for i in range(1, len(time))
    ]

    viols_mean = [
        [np.mean(viols[i][j]) for j in range(len(n_sw))] for i in range(1, len(viols))
    ]

time_max = [[max(time[i][j]) for j in range(len(n_sw))] for i in range(1, len(time))]
time_min = [[min(time[i][j]) for j in range(len(n_sw))] for i in range(1, len(time))]

nodes_max = [[max(nodes[i][j]) for j in range(len(n_sw))] for i in range(1, len(nodes))]

# plot relative error all together
_, axs = plt.subplots(2, 1, constrained_layout=True, sharex=True)
for i, leg_ in enumerate(leg):
    axs[0].plot(5, 5, mf[i], label=leg_, markerfacecolor="none")
axs[0].set_axis_off()
axs[0].legend(leg, ncol=2, loc="center")
axs[1].set_yscale("log")
for i in range(len(types) - 1):
    axs[1].plot(n_sw, perf_drop_rel_mean[i], mf[i])
    axs[1].fill_between(
        n_sw,
        [perf_drop_rel_mean[i][j] - perf_drop_rel_sd[i][j] for j in range(len(n_sw))],
        [perf_drop_rel_mean[i][j] + perf_drop_rel_sd[i][j] for j in range(len(n_sw))],
        alpha=0.2,
    )
axs[1].set_xlabel(r"$n$")
axs[1].set_ylabel(r"$\%J$")

# plot absulute error and time and nodes all together
_, axs = plt.subplots(5, 1, constrained_layout=True, sharex=True)
for i, leg_ in enumerate(leg):
    axs[0].plot(5, 5, mf[i], label=leg_, markerfacecolor="none")
axs[0].set_axis_off()
axs[0].legend(leg, ncol=2, loc="center")
# axs[1].set_yscale("log")
for i in range(len(types) - 1):
    axs[1].plot(n_sw, perf_drop_abs_mean[i], mf[i])
    axs[1].fill_between(
        n_sw,
        [perf_drop_abs_mean[i][j] - perf_drop_abs_sd[i][j] for j in range(len(n_sw))],
        [perf_drop_abs_mean[i][j] + perf_drop_abs_sd[i][j] for j in range(len(n_sw))],
        alpha=0.2,
    )
axs[1].set_ylabel(r"$\Delta J$")
axs[2].set_yscale("log")
for i in range(len(types) - 1):
    axs[2].plot(n_sw, time_mean[i], mf[i])
    axs[2].fill_between(
        n_sw,
        [time_min[i][j] for j in range(len(n_sw))],
        [time_max[i][j] for j in range(len(n_sw))],
        alpha=0.2,
    )
axs[2].set_ylabel(r"$t$")

axs[3].set_yscale("log")
for i in range(len(types) - 1):
    axs[3].plot(n_sw, nodes_max[i], mf[i])
axs[3].set_ylabel(r"$n_{no}$")

# axs[4].set_yscale("log")
for i in range(len(types) - 1):
    axs[4].plot(n_sw, viols_mean[i], mf[i])
    axs[4].fill_between(
        n_sw,
        [viols_mean[i][j] - viols_sd[i][j] for j in range(len(n_sw))],
        [viols_mean[i][j] + viols_sd[i][j] for j in range(len(n_sw))],
        alpha=0.2,
    )
axs[4].set_xlabel(r"$n$")
axs[4].set_ylabel(r"$n_{cv}$")

# plot error and seperated
fig = plt.figure(constrained_layout=True)
gs = GridSpec(6, 2)
axs = [fig.add_subplot(gs[0, 1])]
for i in range(1, 6):
    for j in range(2):
        axs.append(fig.add_subplot(gs[i, j]))
        if i < 5:
            # axs[-1].set_xticks([])
            axs[-1].set_xticklabels([])
axs.append(fig.add_subplot(gs[0, 0]))

# bounds on axes
ub_perf = max(
    [
        max(
            [
                perf_drop_abs_mean[i][j] + perf_drop_abs_sd[i][j]
                for j in range(len(n_sw))
            ]
        )
        for i in range(2 if not multi_lead else 0, len(perf_drop_abs_mean))
    ]
)
lb_perf = min(
    [
        min(
            [
                perf_drop_abs_mean[i][j] - perf_drop_abs_sd[i][j]
                for j in range(len(n_sw))
            ]
        )
        for i in range(len(perf_drop_abs_mean))
    ]
)
ub_time = max(
    [
        max([time_max[i][j] for j in range(len(n_sw))])
        for i in range(len(perf_drop_abs_mean))
    ]
)
lb_time = min(
    [
        min([time_min[i][j] for j in range(len(n_sw))])
        for i in range(len(perf_drop_abs_mean))
    ]
)
ub_node = max(
    [
        max([nodes_max[i][j] for j in range(len(n_sw))])
        for i in range(len(perf_drop_abs_mean))
    ]
)
lb_node = min(
    [
        min([nodes_max[i][j] for j in range(len(n_sw))])
        for i in range(len(perf_drop_abs_mean))
    ]
)
for i in range(1, 5):
    axs[i].set_ylim(lb_perf, ub_perf)
for i in range(5, 9):
    axs[i].set_ylim(lb_time, ub_time)
    axs[i].set_yscale("log")
    axs[i].set_yticks([0.01, 0.1, 1, 10, 100])
    # plt.gca().yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=[1.0]))
for i in range(9, 11):
    axs[i].set_ylim(lb_node, ub_node)
    axs[i].set_yscale("log")

for i, leg_ in enumerate(leg):
    axs[0].plot(5, 5, mf[i], label=leg_, markerfacecolor="none")
axs[0].set_axis_off()
axs[0].legend(leg, ncol=2, loc="center")
for i in range(num_decent_vars):
    axs[1].plot(
        n_sw,
        perf_drop_abs_mean[i],
        mf[i],
        color=f"C{i}",
        linewidth=lw,
        markersize=ms,
        markerfacecolor="none",
    )
    axs[1].fill_between(
        n_sw,
        [perf_drop_abs_mean[i][j] - perf_drop_abs_sd[i][j] for j in range(len(n_sw))],
        [perf_drop_abs_mean[i][j] + perf_drop_abs_sd[i][j] for j in range(len(n_sw))],
        alpha=0.2,
        color=f"C{i}",
    )

    axs[5].plot(
        n_sw,
        time_mean[i],
        mf[i],
        color=f"C{i}",
        linewidth=lw,
        markersize=ms,
        markerfacecolor="none",
    )
    axs[5].fill_between(
        n_sw,
        [time_min[i][j] for j in range(len(n_sw))],
        [time_max[i][j] for j in range(len(n_sw))],
        alpha=0.2,
        color=f"C{i}",
    )

for i in range(num_decent_vars, num_decent_vars + num_seq_vars):
    axs[2].plot(
        n_sw,
        perf_drop_abs_mean[i],
        mf[i],
        color=f"C{i}",
        linewidth=lw,
        markersize=ms,
        markerfacecolor="none",
    )
    axs[2].fill_between(
        n_sw,
        [perf_drop_abs_mean[i][j] - perf_drop_abs_sd[i][j] for j in range(len(n_sw))],
        [perf_drop_abs_mean[i][j] + perf_drop_abs_sd[i][j] for j in range(len(n_sw))],
        alpha=0.2,
        color=f"C{i}",
    )

    axs[6].plot(
        n_sw,
        time_mean[i],
        mf[i],
        color=f"C{i}",
        linewidth=lw,
        markersize=ms,
        markerfacecolor="none",
    )
    axs[6].fill_between(
        n_sw,
        [time_min[i][j] for j in range(len(n_sw))],
        [time_max[i][j] for j in range(len(n_sw))],
        alpha=0.2,
        color=f"C{i}",
    )

for i in range(
    num_decent_vars + num_seq_vars, num_decent_vars + num_seq_vars + num_event_vars
):
    axs[3].plot(
        n_sw,
        perf_drop_abs_mean[i],
        mf[i],
        color=f"C{i}",
        linewidth=lw,
        markersize=ms,
        markerfacecolor="none",
    )
    axs[3].fill_between(
        n_sw,
        [perf_drop_abs_mean[i][j] - perf_drop_abs_sd[i][j] for j in range(len(n_sw))],
        [perf_drop_abs_mean[i][j] + perf_drop_abs_sd[i][j] for j in range(len(n_sw))],
        alpha=0.2,
        color=f"C{i}",
    )

    axs[7].plot(
        n_sw,
        time_mean[i],
        mf[i],
        color=f"C{i}",
        linewidth=lw,
        markersize=ms,
        markerfacecolor="none",
    )
    axs[7].fill_between(
        n_sw,
        [time_min[i][j] for j in range(len(n_sw))],
        [time_max[i][j] for j in range(len(n_sw))],
        alpha=0.2,
        color=f"C{i}",
    )

for i in range(
    num_decent_vars + num_seq_vars + num_event_vars,
    num_decent_vars + num_seq_vars + num_event_vars + num_admm_vars,
):
    axs[4].plot(
        n_sw,
        perf_drop_abs_mean[i],
        mf[i],
        color=f"C{i}",
        linewidth=lw,
        markersize=ms,
        markerfacecolor="none",
    )
    axs[4].fill_between(
        n_sw,
        [perf_drop_abs_mean[i][j] - perf_drop_abs_sd[i][j] for j in range(len(n_sw))],
        [perf_drop_abs_mean[i][j] + perf_drop_abs_sd[i][j] for j in range(len(n_sw))],
        alpha=0.2,
        color=f"C{i}",
    )

    axs[8].plot(
        n_sw,
        time_mean[i],
        mf[i],
        color=f"C{i}",
        linewidth=lw,
        markersize=ms,
        markerfacecolor="none",
    )
    axs[8].fill_between(
        n_sw,
        [time_min[i][j] for j in range(len(n_sw))],
        [time_max[i][j] for j in range(len(n_sw))],
        alpha=0.2,
        color=f"C{i}",
    )

for i in range(num_decent_vars + num_seq_vars + num_event_vars):
    axs[9].plot(
        n_sw,
        nodes_max[i],
        mf[i],
        color=f"C{i}",
        linewidth=lw,
        markersize=ms,
        markerfacecolor="none",
    )
for i in range(
    num_decent_vars + num_seq_vars + num_event_vars,
    num_decent_vars + num_seq_vars + num_event_vars + num_admm_vars,
):
    axs[10].plot(
        n_sw,
        nodes_max[i],
        mf[i],
        color=f"C{i}",
        linewidth=lw,
        markersize=ms,
        markerfacecolor="none",
    )

for i in range(num_decent_vars):
    axs[-1].plot(
        n_sw,
        perf_drop_abs_mean[i],
        mf[i],
        color=f"C{i}",
        linewidth=lw,
        markersize=ms,
        markerfacecolor="none",
    )
    axs[-1].fill_between(
        n_sw,
        [perf_drop_abs_mean[i][j] - perf_drop_abs_sd[i][j] for j in range(len(n_sw))],
        [perf_drop_abs_mean[i][j] + perf_drop_abs_sd[i][j] for j in range(len(n_sw))],
        alpha=0.2,
        color=f"C{i}",
    )
axs[9].set_xlabel(r"$n$")
axs[10].set_xlabel(r"$n$")
axs[9].set_ylabel(r"$n_{no}$")
axs[1].set_ylabel(r"$\Delta J$")
axs[3].set_ylabel(r"$\Delta J$")
axs[5].set_ylabel(r"$t$")
axs[7].set_ylabel(r"$t$")
plt.show()
