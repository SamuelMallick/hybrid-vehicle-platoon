import pickle

import matplotlib.pyplot as plt
import numpy as np

from examples.ACC_fleet.ACC_model import ACC

plt.rc("text", usetex=True)
plt.rc("font", size=14)
plt.style.use("bmh")

nx_l = 2
plot_len = 50
name = "cent"
DG = False
Q = True
HOM = True
n = 4
N = 4
LT = 1
with open(
    f"{name}_n_{n}_N_{N}_Q_{Q}_DG_{DG}_HOM_{HOM}_LT_{LT}.pkl",
    "rb",
) as file:
    X_cent = pickle.load(file)
    U_cent = pickle.load(file)
    R_cent = pickle.load(file)
    solve_times = pickle.load(file)
    node_counts = pickle.load(file)
    violations = pickle.load(file)
    leader_state = pickle.load(file)
name = "event10"
with open(
    f"{name}_n_{n}_N_{N}_Q_{Q}_DG_{DG}_HOM_{HOM}_LT_{LT}.pkl",
    "rb",
) as file:
    X_seq = pickle.load(file)
    U_seq = pickle.load(file)
    R_seq = pickle.load(file)
    solve_times = pickle.load(file)
    node_counts = pickle.load(file)
    violations = pickle.load(file)
    leader_state = pickle.load(file)

# plt.plot(X_cent[:, [0, 2, 4, 6, 8]], 'r')
# plt.plot(X_seq[:, [0, 2, 4, 6, 8]], 'b')
# plt.plot(X_cent[:, [1, 3, 5, 7, 9]], 'r')
# plt.plot(X_seq[:, [1, 3, 5, 7, 9]], 'b')
plt.plot(R_cent.squeeze())
plt.plot(R_seq.squeeze())
print(f"cent cost: {sum(R_cent)}")
print(f"seq cost: {sum(R_seq)}")

acc = ACC(plot_len, N, leader_traj=LT)
leader_state = acc.get_leader_state()
nx_l = acc.nx_l
nu_l = acc.nu_l
Q_x_l = acc.Q_x_l
Q_u_l = acc.Q_u_l
Q_du_l = acc.Q_du_l
sep = acc.sep

cost = 0
cost_2 = 0
X = X_cent
U = U_cent
for k in range(plot_len):
    for i in range(n):
        local_state = X[[k], nx_l * i : nx_l * (i + 1)]
        local_control = U[[k], nu_l * i : nu_l * (i + 1)]
        if i == 0:
            # first car follows traj with no sep
            follow_state = leader_state[:, k]
            temp_sep = np.zeros((2, 1))
        else:
            # otherwise follow car infront (i-1)
            follow_state = X[[k], nx_l * (i - 1) : nx_l * (i)]
            temp_sep = sep
        cost += (local_state - follow_state - temp_sep.T) @ Q_x_l @ (
            local_state - follow_state - temp_sep.T
        ).T + local_control @ Q_u_l @ local_control.T

plt.show()
