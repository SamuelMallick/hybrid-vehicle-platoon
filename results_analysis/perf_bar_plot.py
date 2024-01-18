import pickle

import matplotlib.pyplot as plt

from dmpcpwa.utils.tikz import save2tikz

plt.rc("text", usetex=True)
plt.rc("font", size=14)
plt.style.use("bmh")

nx_l = 2
LT = 2
HOM = True
DG = False
Q = True
N = 5
n = 5
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
track_costs = []
for type in types:
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

        track_costs.append(sum(R))

perf_drop = []
for i in range(1, len(types)):
    perf_drop.append(
        100 * (track_costs[i][0, 0] - track_costs[0][0, 0]) / track_costs[0][0, 0]
    )

plt.bar(
    [i for i in range(1, len(types))],
    perf_drop,
    tick_label=leg,
    color=[f"C{i-1}" for i in range(1, len(types))],
)
plt.yscale("log")
plt.ylabel(r"$\%J$")
save2tikz(plt.gcf())
plt.show()
