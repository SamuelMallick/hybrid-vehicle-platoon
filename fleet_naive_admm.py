import pickle
import sys

import gurobipy as gp
import matplotlib.pyplot as plt
import numpy as np
from dmpcpwa.agents.mld_agent import MldAgent
from dmpcpwa.mpc.mpc_mld import MpcMld
from dmpcrl.core.admm import g_map
from gymnasium import Env
from gymnasium.wrappers import TimeLimit
from mpcrl.wrappers.envs import MonitorEpisodes

from env import CarFleet
from models import ACC
from mpcs.cent_mld import MPCMldCent
from mpcs.mpc_gear import MpcGear
from plot_fleet import plot_fleet

np.random.seed(2)

PLOT = False
SAVE = True

DEBUG_PLOT = False  # when true, the admm iterations are plotted at each time step

n = 3  # num cars
N = 5  # controller horizon
COST_2_NORM = True
DISCRETE_GEARS = False
HOMOGENOUS = True
LEADER_TRAJ = 2  # "1" - constant velocity leader traj. Vehicles start from random ICs. "2" - accelerating leader traj. Vehicles start in perfect platoon.

admm_iters = 50  # fixed number of iterations for ADMM routine
if len(sys.argv) > 1:
    n = int(sys.argv[1])
if len(sys.argv) > 2:
    N = int(sys.argv[2])
if len(sys.argv) > 3:
    COST_2_NORM = bool(int(sys.argv[3]))
if len(sys.argv) > 4:
    DISCRETE_GEARS = bool(int(sys.argv[4]))
if len(sys.argv) > 5:
    HOMOGENOUS = bool(int(sys.argv[5]))
if len(sys.argv) > 6:
    LEADER_TRAJ = int(sys.argv[6])
if len(sys.argv) > 7:
    admm_iters = int(sys.argv[7])

random_ICs = False
if LEADER_TRAJ == 1:
    random_ICs = True

ep_len = 100  # length of episode (sim len)
Adj = np.zeros((n, n))  # adjacency matrix
if n > 1:
    for i in range(n):  # make it chain coupling
        if i == 0:
            Adj[i, i + 1] = 1
        elif i == n - 1:
            Adj[i, i - 1] = 1
        else:
            Adj[i, i + 1] = 1
            Adj[i, i - 1] = 1
else:
    Adj = np.zeros((1, 1))
G_map = g_map(Adj)

acc = ACC(ep_len, N, leader_traj=LEADER_TRAJ)
nx_l = acc.nx_l
nu_l = acc.nu_l
Q_x_l = acc.Q_x_l
Q_u_l = acc.Q_u_l
Q_du_l = acc.Q_du_l
sep = acc.sep
d_safe = acc.d_safe
w = acc.w  # slack variable penalty
leader_state = acc.get_leader_state()

large_num = 100000  # large number for dumby bounds on vars
rho = 0.5  # admm penalty


class LocalMpcADMM(MpcMld):
    def __init__(
        self, system: dict, N: int, leader: bool = False, trailer: bool = False
    ) -> None:
        super().__init__(system, N)
        self.setup_cost_and_constraints(self.u, leader, trailer)

    def setup_cost_and_constraints(self, u, leader=False, trailer=False):
        if COST_2_NORM:
            cost_func = self.min_2_norm
        else:
            cost_func = self.min_1_norm

        # vars for front and back car
        self.x_front = self.mpc_model.addMVar(
            (nx_l, N + 1), lb=-large_num, ub=large_num, name="x_front"
        )
        self.x_back = self.mpc_model.addMVar(
            (nx_l, N + 1), lb=-large_num, ub=large_num, name="x_back"
        )

        self.s_front = self.mpc_model.addMVar(
            (1, N + 1), lb=0, ub=float("inf"), name="s_front"
        )

        self.s_back = self.mpc_model.addMVar(
            (1, N + 1), lb=0, ub=float("inf"), name="s_back"
        )

        # admm vars
        self.y_front = self.mpc_model.addMVar((nx_l, N + 1), lb=0, ub=0, name="y_front")

        self.y_back = self.mpc_model.addMVar((nx_l, N + 1), lb=0, ub=0, name="y_back")

        self.z_front = self.mpc_model.addMVar((nx_l, N + 1), lb=0, ub=0, name="z_front")

        self.z_back = self.mpc_model.addMVar((nx_l, N + 1), lb=0, ub=0, name="z_back")

        # setting these bounds to zero removes the slack var, as leader and trailer
        # dont have cars in front or behind respectively
        if leader:
            self.s_front.ub = 0
        if trailer:
            self.s_back.ub = 0

        obj = 0
        if leader:
            temp_sep = np.zeros((2, 1))
        else:
            temp_sep = sep

        for k in range(N):
            obj += cost_func(self.x[:, [k]] - self.x_front[:, [k]] - temp_sep, Q_x_l)
            obj += (
                cost_func(u[:, [k]], Q_u_l)
                + w * self.s_front[:, k]
                + w * self.s_back[:, k]
            )

            if k < N - 1:
                obj += cost_func(u[:, [k + 1]] - u[:, [k]], Q_du_l)

            # accel cnstrs
            self.mpc_model.addConstr(
                self.x[1, [k + 1]] - self.x[1, [k]] <= acc.a_acc * acc.ts,
                name=f"acc_{k}",
            )
            self.mpc_model.addConstr(
                self.x[1, [k + 1]] - self.x[1, [k]] >= acc.a_dec * acc.ts,
                name=f"dec_{k}",
            )
            # safe distance constraints and admm terms
            if not leader:
                self.mpc_model.addConstr(
                    self.x[0, [k]] - self.s_front[:, [k]]
                    <= self.x_front[0, [k]] - d_safe,
                    name=f"safety_ahead_{k}",
                )

                obj += self.y_front[:, k] @ (
                    self.x_front[:, [k]] - self.z_front[:, [k]]
                )
                obj += (
                    (rho / 2)
                    * (self.x_front[:, k] - self.z_front[:, k])
                    @ np.eye(nx_l)
                    @ (self.x_front[:, [k]] - self.z_front[:, [k]])
                )

            if not trailer:
                self.mpc_model.addConstr(
                    self.x[0, [k]] + self.s_back[:, [k]]
                    >= self.x_back[0, [k]] + d_safe,
                    name=f"safety_behind_{k}",
                )

                obj += self.y_back[:, k] @ (self.x_back[:, [k]] - self.z_back[:, [k]])
                obj += (
                    (rho / 2)
                    * (self.x_back[:, k] - self.z_back[:, k])
                    @ np.eye(nx_l)
                    @ (self.x_back[:, [k]] - self.z_back[:, [k]])
                )

        obj += cost_func(self.x[:, [N]] - self.x_front[:, [N]] - temp_sep, Q_x_l)
        obj += +w * self.s_front[:, N] + w * self.s_back[:, N]

        if not leader:
            self.mpc_model.addConstr(
                self.x[0, [N]] - self.s_front[:, [N]] <= self.x_front[0, [N]] - d_safe,
                name=f"safety_ahead_{N}",
            )

            obj += self.y_front[:, N] @ (self.x_front[:, [N]] - self.z_front[:, [N]])
            obj += (
                (rho / 2)
                * (self.x_front[:, N] - self.z_front[:, N])
                @ np.eye(nx_l)
                @ (self.x_front[:, [N]] - self.z_front[:, [N]])
            )

        if not trailer:
            self.mpc_model.addConstr(
                self.x[0, [N]] + self.s_back[:, [N]] >= self.x_back[0, [N]] + d_safe,
                name=f"safety_behind_{N}",
            )

            obj += self.y_back[:, N] @ (self.x_back[:, [N]] - self.z_back[:, [N]])
            obj += (
                (rho / 2)
                * (self.x_back[:, N] - self.z_back[:, N])
                @ np.eye(nx_l)
                @ (self.x_back[:, [N]] - self.z_back[:, [N]])
            )

        self.mpc_model.setObjective(obj, gp.GRB.MINIMIZE)

    def set_front_vars(self, y_front, z_front):
        for k in range(N + 1):
            self.y_front[:, [k]].lb = y_front[:, [k]]
            self.y_front[:, [k]].ub = y_front[:, [k]]

            self.z_front[:, [k]].lb = z_front[:, [k]]
            self.z_front[:, [k]].ub = z_front[:, [k]]

    def set_back_vars(self, y_back, z_back):
        for k in range(N + 1):
            self.y_back[:, [k]].lb = y_back[:, [k]]
            self.y_back[:, [k]].ub = y_back[:, [k]]

            self.z_back[:, [k]].lb = z_back[:, [k]]
            self.z_back[:, [k]].ub = z_back[:, [k]]

    def set_x_front(self, x_front):
        for k in range(N + 1):
            self.x_front[:, [k]].lb = x_front[:, [k]]
            self.x_front[:, [k]].ub = x_front[:, [k]]


class LocalMpcGear(LocalMpcADMM, MpcGear):
    def __init__(
        self, system: dict, N: int, leader: bool = False, trailer: bool = False
    ) -> None:
        MpcGear.__init__(self, system, N)
        self.setup_gears(N, acc, system["F"], system["G"])
        self.setup_cost_and_constraints(self.u_g, leader, trailer)


class ADMMCoordinator(MldAgent):
    def __init__(
        self,
        local_mpcs: list[MpcMld],
    ) -> None:
        """Initialise the coordinator.

        Parameters
        ----------
        local_mpcs: List[MpcMld]
            List of local MLD based MPCs - one for each agent.
        """
        super().__init__(local_mpcs[0])  # just pass first mpc to satisfy constructor

        self.n = len(local_mpcs)
        self.agents: list[MldAgent] = []
        for i in range(self.n):
            self.agents.append(MldAgent(local_mpcs[i]))

        # admm vars
        self.y_front_list = [np.zeros((nx_l, N + 1)) for i in range(self.n)]
        self.y_back_list = [np.zeros((nx_l, N + 1)) for i in range(self.n)]
        self.z_list = [np.zeros((nx_l, N + 1)) for i in range(self.n)]

        self.solve_times = np.zeros((ep_len, 1))
        self.node_counts = np.zeros((ep_len, 1))
        # temp value used to sum up the itermediate solves over iterations
        self.temp_solve_time = 0
        self.temp_node_count = 0

        # a centralized mpc to generate global optimums, to analyse admm convergence
        if DEBUG_PLOT:
            if DISCRETE_GEARS:
                raise RuntimeError(
                    "Admm debug with centralized mpc not implemented for discrete gear model."
                )
            if HOMOGENOUS:  # by not passing the index all systems are the same
                systems = [acc.get_pwa_system() for i in range(n)]
            else:
                systems = [acc.get_pwa_system(i) for i in range(n)]
            self.cent_mpc = MPCMldCent(systems, acc, COST_2_NORM, n, N)

    def get_control(self, state):
        u = [None] * self.n
        if DEBUG_PLOT:
            admm_dict = {
                "u": [[] for i in range(n)],
                "x": [[] for i in range(n)],
                "x_front": [[] for i in range(n)],
                "x_back": [[] for i in range(n)],
                "z": [[] for i in range(n)],
            }

        # initial guess for coupling vars in admm comes from previous solutions #TODO timeshift with constant vel
        for i in range(self.n):
            if i != 0:  # first car has no car in front
                x_pred_ahead = self.agents[i - 1].get_predicted_state(shifted=True)
                if x_pred_ahead is not None:  # will be none on first time-step
                    self.agents[i].mpc.set_front_vars(
                        self.y_front_list[i], x_pred_ahead
                    )
            if i != n - 1:  # last car has no car behind
                x_pred_behind = self.agents[i + 1].get_predicted_state(shifted=True)
                if x_pred_behind is not None:  # will be none on first time-step
                    self.agents[i].mpc.set_back_vars(self.y_back_list[i], x_pred_behind)

        for t in range(admm_iters):
            # admm x-update
            for i in range(self.n):
                xl = state[nx_l * i : nx_l * (i + 1), :]  # pull out local part of state
                u[i] = self.agents[i].get_control(xl)
                if DEBUG_PLOT:
                    admm_dict["u"][i].append(self.agents[i].mpc.u.X)
                    admm_dict["x"][i].append(self.agents[i].mpc.x.X)
                    if i != 0:
                        admm_dict["x_front"][i].append(self.agents[i].mpc.x_front.X)
                    if i != n - 1:
                        admm_dict["x_back"][i].append(self.agents[i].mpc.x_back.X)

            # admm z-update and y-update together
            for i in range(self.n):
                if i == 0:
                    self.z_list[i] = (1.0 / 2.0) * (
                        self.agents[i].mpc.x.X + self.agents[i + 1].mpc.x_front.X
                    )
                    self.y_front_list[i + 1] += rho * (
                        self.agents[i + 1].mpc.x_front.X - self.z_list[i]
                    )
                elif i == n - 1:
                    self.z_list[i] = (1.0 / 2.0) * (
                        self.agents[i].mpc.x.X + self.agents[i - 1].mpc.x_back.X
                    )
                    self.y_back_list[i - 1] += rho * (
                        self.agents[i - 1].mpc.x_back.X - self.z_list[i]
                    )
                else:
                    self.z_list[i] = (1.0 / 3.0) * (
                        self.agents[i].mpc.x.X
                        + self.agents[i + 1].mpc.x_front.X
                        + self.agents[i - 1].mpc.x_back.X
                    )
                    self.y_front_list[i + 1] += rho * (
                        self.agents[i + 1].mpc.x_front.X - self.z_list[i]
                    )
                    self.y_back_list[i - 1] += rho * (
                        self.agents[i - 1].mpc.x_back.X - self.z_list[i]
                    )

                if DEBUG_PLOT:
                    admm_dict["z"][i].append(self.z_list[i])

            # update z and y for local agents
            for i in range(n):
                if i == 0:
                    self.agents[i].mpc.set_back_vars(
                        self.y_back_list[i], self.z_list[i + 1]
                    )
                elif i == n - 1:
                    self.agents[i].mpc.set_front_vars(
                        self.y_front_list[i], self.z_list[i - 1]
                    )
                else:
                    self.agents[i].mpc.set_front_vars(
                        self.y_front_list[i], self.z_list[i - 1]
                    )
                    self.agents[i].mpc.set_back_vars(
                        self.y_back_list[i], self.z_list[i + 1]
                    )

            # get solve times and node count
            self.temp_solve_time += max(
                [self.agents[i].run_time for i in range(self.n)]
            )
            self.temp_node_count = max(
                max([self.agents[i].node_count for i in range(self.n)]),
                self.temp_node_count,
            )

        if DEBUG_PLOT:
            tot_resid = 0
            for i in range(n):
                tot_resid += sum(
                    np.linalg.norm(
                        admm_dict["x"][i][-1][:, time_step]
                        - admm_dict["z"][i][-1][:, time_step]
                    )
                    for time_step in range(N)
                )
                if i != 0:
                    tot_resid += sum(
                        np.linalg.norm(
                            admm_dict["x_back"][i - 1][-1][:, time_step]
                            - admm_dict["z"][i][-1][:, time_step]
                        )
                        for time_step in range(N)
                    )
                if i != n - 1:
                    tot_resid += sum(
                        np.linalg.norm(
                            admm_dict["x_front"][i + 1][-1][:, time_step]
                            - admm_dict["z"][i][-1][:, time_step]
                        )
                        for time_step in range(N)
                    )
            print(f"Total residual = {tot_resid}")
            # centralized solution
            self.cent_mpc.solve_mpc(state)
            time_step = 0  # plot control/state iterations predicted at this time_step
            agent = 2  # plot for which agent
            # plt.plot([admm_dict["u"][agent][i][:, time_step] for i in range(admm_iters)])
            for time_step in range(N):
                plt.plot(
                    [admm_dict["u"][agent][i][:, time_step] for i in range(admm_iters)]
                )
                plt.axhline(self.cent_mpc.u.X[agent, time_step], linestyle="--")
                plt.show()

                for a in range(n):
                    plt.plot(
                        [
                            admm_dict["x"][a][i][0, time_step]
                            - admm_dict["z"][a][i][0, time_step]
                            for i in range(admm_iters)
                        ]
                    )
                plt.title("Residual")
                plt.show()

                plt.plot(
                    [admm_dict["x"][agent][i][0, time_step] for i in range(admm_iters)]
                )
                if agent != 0:
                    plt.plot(
                        [
                            admm_dict["x_back"][agent - 1][i][0, time_step]
                            for i in range(admm_iters)
                        ]
                    )
                if agent != n - 1:
                    plt.plot(
                        [
                            admm_dict["x_front"][agent + 1][i][0, time_step]
                            for i in range(admm_iters)
                        ]
                    )
                plt.axhline(self.cent_mpc.x.X[agent * nx_l, time_step], linestyle="--")
                plt.show()

        if DISCRETE_GEARS:
            # stack the continuous conttrol at the front and the discrete at the back
            return np.vstack(
                (
                    np.vstack([u[i][:nu_l, :] for i in range(n)]),
                    np.vstack([u[i][nu_l:, :] for i in range(n)]),
                )
            )
        else:
            return np.vstack(u)

    # here we set the leader cost because it is independent of other vehicles' states
    def on_timestep_end(self, env: Env, episode: int, timestep: int) -> None:
        x_goal = leader_state[:, timestep : timestep + N + 1]
        self.agents[0].mpc.set_x_front(x_goal)

        # set the leader trajectory for centralized MPC
        if DEBUG_PLOT:
            self.cent_mpc.set_leader_traj(x_goal)

        self.solve_times[env.step_counter - 1, :] = self.temp_solve_time
        self.node_counts[env.step_counter - 1, :] = self.temp_node_count
        self.temp_solve_time = 0
        self.temp_node_count = 0

        return super().on_timestep_end(env, episode, timestep)

    def on_episode_start(self, env: Env, episode: int, state) -> None:
        x_goal = leader_state[:, 0 : N + 1]
        self.agents[0].mpc.set_x_front(x_goal)
        return super().on_episode_start(env, episode, state)


# env
env = MonitorEpisodes(
    TimeLimit(
        CarFleet(
            acc,
            n,
            ep_len,
            L2_norm_cost=COST_2_NORM,
            homogenous=HOMOGENOUS,
            random_ICs=random_ICs,
        ),
        max_episode_steps=ep_len,
    )
)

if DISCRETE_GEARS:
    mpc_class = LocalMpcGear
    if HOMOGENOUS:  # by not passing the index all systems are the same
        systems = [acc.get_friction_pwa_system() for i in range(n)]
    else:
        systems = [acc.get_friction_pwa_system(i) for i in range(n)]
else:
    mpc_class = LocalMpcADMM
    if HOMOGENOUS:  # by not passing the index all systems are the same
        systems = [acc.get_pwa_system() for i in range(n)]
    else:
        systems = [acc.get_pwa_system(i) for i in range(n)]
# coordinator
local_mpcs: list[MpcMld] = []
for i in range(n):
    # passing local system
    if i == 0:
        local_mpcs.append(mpc_class(systems[i], N, leader=True))
    elif i == n - 1:
        local_mpcs.append(mpc_class(systems[i], N, trailer=True))
    else:
        local_mpcs.append(mpc_class(systems[i], N))
agent = ADMMCoordinator(local_mpcs)

agent.evaluate(env=env, episodes=1, seed=1)

if len(env.observations) > 0:
    X = env.observations[0].squeeze()
    U = env.actions[0].squeeze()
    R = env.rewards[0]
else:
    X = np.squeeze(env.ep_observations)
    U = np.squeeze(env.ep_actions)
    R = np.squeeze(env.ep_rewards)

print(f"Return = {sum(R.squeeze())}")
print(f"Violations = {env.unwrapped.viol_counter}")
print(f"Run_times_sum: {sum(agent.solve_times)}")

if PLOT:
    plot_fleet(n, X, U, R, leader_state, violations=env.unwrapped.viol_counter[0])

if SAVE:
    with open(
        f"admm{admm_iters}_n_{n}_N_{N}_Q_{COST_2_NORM}_DG_{DISCRETE_GEARS}_HOM_{HOMOGENOUS}_LT_{LEADER_TRAJ}"
        # + datetime.datetime.now().strftime("%d%H%M%S%f")
        + ".pkl",
        "wb",
    ) as file:
        pickle.dump(X, file)
        pickle.dump(U, file)
        pickle.dump(R, file)
        pickle.dump(agent.solve_times, file)
        pickle.dump(agent.node_counts, file)
        pickle.dump(env.unwrapped.viol_counter[0], file)
        pickle.dump(leader_state, file)
