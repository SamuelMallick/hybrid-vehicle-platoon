import matplotlib.pyplot as plt

plt.rc("text", usetex=True)
plt.rc("font", size=14)
plt.style.use("bmh")

nx_l = 2


def plot_fleet(n, X, U, R, leader_state, violations=None):
    _, axs = plt.subplots(2, 1, constrained_layout=True, sharex=True)
    axs[0].plot(leader_state[0, :], "--")
    axs[1].plot(leader_state[1, :], "--")
    for i in range(n):
        axs[0].plot(X[:, nx_l * i])
        axs[1].plot(X[:, nx_l * i + 1])
    axs[0].set_ylabel(f"pos (m)")
    axs[1].set_ylabel("vel (ms-1)")
    axs[1].set_xlabel(f"time step k")
    axs[0].legend(["reference"])
    # if violations is not None:
    #    axs[0].plot(violations)
    _, axs = plt.subplots(1, 1, constrained_layout=True, sharex=True)
    axs.plot(U)
    _, axs = plt.subplots(1, 1, constrained_layout=True, sharex=True)
    axs.plot(R.squeeze())
    plt.show()
