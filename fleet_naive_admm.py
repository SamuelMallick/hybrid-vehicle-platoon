import pickle

import gurobipy as gp
import matplotlib.pyplot as plt
import numpy as np
from dmpcpwa.agents.mld_agent import MldAgent
from dmpcpwa.mpc.mpc_mld import MpcMld
from dmpcrl.core.admm import g_map
from gymnasium import Env
from gymnasium.wrappers import TimeLimit
from mpcrl.wrappers.envs import MonitorEpisodes

from env import PlatoonEnv
from misc.common_controller_params import Params, Sim
from misc.spacing_policy import ConstantSpacingPolicy, SpacingPolicy
from models import Platoon, Vehicle
from mpcs.mpc_gear import MpcGear
from plot_fleet import plot_fleet

np.random.seed(2)

DEBUG_PLOT = False  # when true, the admm iterations are plotted at each time step


class LocalMpcADMM(MpcMld):
    """A local MPC in the ADMM scheme for the platoon"""

    Q_x = Params.Q_x
    Q_u = Params.Q_u
    Q_du = Params.Q_du
    w = Params.w
    a_acc = Params.a_acc
    a_dec = Params.a_dec
    ts = Params.ts
    d_safe = Params.d_safe

    def __init__(
        self,
        N: int,
        pwa_system: dict,
        rho: float,
        spacing_policy: SpacingPolicy = ConstantSpacingPolicy(50),
        quadratic_cost: bool = True,
        is_leader: bool = False,
        is_trailer: bool = False,
    ) -> None:
        super().__init__(pwa_system, N)
        self.rho = rho
        self.setup_cost_and_constraints(
            self.u, spacing_policy, quadratic_cost, is_leader, is_trailer
        )

    def setup_cost_and_constraints(
        self,
        u,
        spacing_policy: SpacingPolicy = ConstantSpacingPolicy(50),
        quadratic_cost: bool = True,
        is_leader=False,
        is_trailer=False,
    ):
        """Set up  cost and constraints for vehicle. Penalises the u passed in."""
        if quadratic_cost:
            self.cost_func = self.min_2_norm
        else:
            self.cost_func = self.min_1_norm

        nx_l = Vehicle.nx_l
        Vehicle.nu_l

        # copies of states for prec and succ vehicle
        self.x_front = self.mpc_model.addMVar(
            (nx_l, self.N + 1), lb=-1e6, ub=1e6, name="x_front"
        )
        self.x_back = self.mpc_model.addMVar(
            (nx_l, self.N + 1), lb=-1e6, ub=1e6, name="x_back"
        )
        # slack vars for constraints with prec and succ vehicles
        self.s_front = self.mpc_model.addMVar(
            (self.N + 1), lb=0, ub=float("inf"), name="s_front"
        )
        self.s_back = self.mpc_model.addMVar(
            (self.N + 1), lb=0, ub=float("inf"), name="s_back"
        )

        # admm vars
        self.y_front = self.mpc_model.addMVar(
            (nx_l, self.N + 1), lb=0, ub=0, name="y_front"
        )
        self.y_back = self.mpc_model.addMVar(
            (nx_l, self.N + 1), lb=0, ub=0, name="y_back"
        )
        self.z_front = self.mpc_model.addMVar(
            (nx_l, self.N + 1), lb=0, ub=0, name="z_front"
        )
        self.z_back = self.mpc_model.addMVar(
            (nx_l, self.N + 1), lb=0, ub=0, name="z_back"
        )

        # setting these bounds to zero removes the slack var, as leader and trailer
        # dont have cars in front or behind respectively
        if is_leader:
            self.s_front.ub = 0
        if is_trailer:
            self.s_back.ub = 0

        # cost func
        cost = 0
        # tracking cost
        if is_leader:
            cost += sum(
                [
                    self.cost_func(self.x[:, [k]] - self.x_front[:, [k]], self.Q_x)
                    for k in range(self.N + 1)
                ]
            )
        else:
            cost += sum(
                [
                    self.cost_func(
                        self.x[:, [k]]
                        - self.x_front[:, [k]]
                        - spacing_policy.spacing(self.x[:, [k]]),
                        self.Q_x,
                    )
                    for k in range(self.N + 1)
                ]
            )
        # control effort cost
        cost += sum([self.cost_func(u[:, [k]], self.Q_u) for k in range(self.N)])
        # control variation cost
        cost += sum(
            [
                self.cost_func(u[:, [k + 1]] - u[:, [k]], self.Q_du)
                for k in range(self.N - 1)
            ]
        )
        # slack variable cost
        cost += sum(
            [
                self.w * self.s_front[k] + self.w * self.s_back[k]
                for k in range(self.N + 1)
            ]
        )
        # admm cost
        if not is_leader:
            cost += sum(
                [
                    self.y_front[:, k] @ (self.x_front[:, [k]] - self.z_front[:, [k]])
                    + (
                        (self.rho / 2)
                        * (self.x_front[:, k] - self.z_front[:, k])
                        @ np.eye(nx_l)
                        @ (self.x_front[:, [k]] - self.z_front[:, [k]])
                    )
                    for k in range(self.N + 1)
                ]
            )
        if not is_trailer:
            cost += sum(
                [
                    self.y_back[:, k] @ (self.x_back[:, [k]] - self.z_back[:, [k]])
                    + (
                        (self.rho / 2)
                        * (self.x_back[:, k] - self.z_back[:, k])
                        @ np.eye(nx_l)
                        @ (self.x_back[:, [k]] - self.z_back[:, [k]])
                    )
                    for k in range(self.N + 1)
                ]
            )
        self.mpc_model.setObjective(cost, gp.GRB.MINIMIZE)

        # accel constraints
        self.mpc_model.addConstrs(
            (
                self.a_dec * self.ts <= self.x[1, [k + 1]] - self.x[1, [k]]
                for k in range(self.N)
            ),
            name="dec",
        )
        self.mpc_model.addConstrs(
            (
                self.x[1, [k + 1]] - self.x[1, [k]] <= self.a_acc * self.ts
                for k in range(self.N)
            ),
            name="acc",
        )

        # safe distance constraints
        if not is_leader:
            self.mpc_model.addConstrs(
                (
                    self.x[0, [k]] - self.s_front[[k]]
                    <= self.x_front[0, [k]] - self.d_safe
                    for k in range(self.N + 1)
                ),
                name="safe_front",
            )
        if not is_trailer:
            self.mpc_model.addConstrs(
                (
                    self.x[0, [k]] + self.s_back[[k]]
                    >= self.x_back[0, [k]] + self.d_safe
                    for k in range(self.N + 1)
                ),
                name="safe_back",
            )

    def set_front_vars(self, y_front, z_front):
        for k in range(self.N + 1):
            self.y_front[:, [k]].lb = y_front[:, [k]]
            self.y_front[:, [k]].ub = y_front[:, [k]]

            self.z_front[:, [k]].lb = z_front[:, [k]]
            self.z_front[:, [k]].ub = z_front[:, [k]]

    def set_back_vars(self, y_back, z_back):
        for k in range(self.N + 1):
            self.y_back[:, [k]].lb = y_back[:, [k]]
            self.y_back[:, [k]].ub = y_back[:, [k]]

            self.z_back[:, [k]].lb = z_back[:, [k]]
            self.z_back[:, [k]].ub = z_back[:, [k]]

    def set_x_front(self, x_front):
        for k in range(self.N + 1):
            self.x_front[:, [k]].lb = x_front[:, [k]]
            self.x_front[:, [k]].ub = x_front[:, [k]]


class LocalMpcGear(LocalMpcADMM, MpcGear):
    def __init__(
        self,
        N: int,
        system: dict,
        rho: float,
        spacing_policy: SpacingPolicy = ConstantSpacingPolicy(50),
        quadratic_cost: bool = True,
        is_leader: bool = False,
        is_trailer: bool = False,
    ) -> None:
        MpcGear.__init__(self, system, N)
        self.rho = rho
        self.setup_gears(N, system["F"], system["G"])
        self.setup_cost_and_constraints(
            self.u_g, spacing_policy, quadratic_cost, is_leader, is_trailer
        )


class ADMMCoordinator(MldAgent):
    def __init__(
        self,
        local_mpcs: list[MpcMld],
        admm_iters: int,
        ep_len: int,
        N: int,
        leader_x: np.ndarray,
        ts: float,
        rho: float,
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

        self.leader_x = leader_x
        self.nx_l = Vehicle.nx_l
        self.nu_l = Vehicle.nu_l
        self.ep_len = ep_len
        self.ts = ts
        self.N = N
        self.admm_iters = admm_iters
        self.rho = rho

        # admm vars
        self.y_front_list = [np.zeros((self.nx_l, N + 1)) for i in range(self.n)]
        self.y_back_list = [np.zeros((self.nx_l, N + 1)) for i in range(self.n)]
        self.z_list = [np.zeros((self.nx_l, N + 1)) for i in range(self.n)]

        self.solve_times = np.zeros((ep_len, 1))
        self.node_counts = np.zeros((ep_len, 1))
        # temp value used to sum up the itermediate solves over iterations
        self.temp_solve_time = 0
        self.temp_node_count = 0

        # a centralized mpc to generate global optimums, to analyse admm convergence
        # if DEBUG_PLOT:
        #     if DISCRETE_GEARS:
        #         raise RuntimeError(
        #             "Admm debug with centralized mpc not implemented for discrete gear model."
        #         )
        #     if HOMOGENOUS:  # by not passing the index all systems are the same
        #         systems = [acc.get_pwa_system() for i in range(n)]
        #     else:
        #         systems = [acc.get_pwa_system(i) for i in range(n)]
        #     self.cent_mpc = MPCMldCent(systems, acc, COST_2_NORM, n, N)

    def get_control(self, state):
        x_l = np.split(state, self.n, axis=0)  # split into local state components
        u = [None] * self.n
        if DEBUG_PLOT:
            admm_dict = {
                "u": [[] for i in range(self.n)],
                "x": [[] for i in range(self.n)],
                "x_front": [[] for i in range(self.n)],
                "x_back": [[] for i in range(self.n)],
                "z": [[] for i in range(self.n)],
            }

        # initial guess for coupling vars in admm comes from previous solutions
        for i in range(self.n):
            if i != 0:  # first car has no car in front
                x_pred_ahead = self.agents[i - 1].get_predicted_state(shifted=True)
                if x_pred_ahead is not None:  # will be none on first time-step
                    self.agents[i].mpc.set_front_vars(
                        self.y_front_list[i], x_pred_ahead
                    )
            if i != self.n - 1:  # last car has no car behind
                x_pred_behind = self.agents[i + 1].get_predicted_state(shifted=True)
                if x_pred_behind is not None:  # will be none on first time-step
                    self.agents[i].mpc.set_back_vars(self.y_back_list[i], x_pred_behind)

        for t in range(self.admm_iters):
            # admm x-update
            for i in range(self.n):
                xl = state[
                    self.nx_l * i : self.nx_l * (i + 1), :
                ]  # pull out local part of state
                u[i] = self.agents[i].get_control(xl)
                if DEBUG_PLOT:
                    admm_dict["u"][i].append(self.agents[i].mpc.u.X)
                    admm_dict["x"][i].append(self.agents[i].mpc.x.X)
                    if i != 0:
                        admm_dict["x_front"][i].append(self.agents[i].mpc.x_front.X)
                    if i != self.n - 1:
                        admm_dict["x_back"][i].append(self.agents[i].mpc.x_back.X)

            # admm z-update and y-update together
            for i in range(self.n):
                if i == 0:
                    self.z_list[i] = (1.0 / 2.0) * (
                        self.agents[i].mpc.x.X + self.agents[i + 1].mpc.x_front.X
                    )
                    self.y_front_list[i + 1] += self.rho * (
                        self.agents[i + 1].mpc.x_front.X - self.z_list[i]
                    )
                elif i == self.n - 1:
                    self.z_list[i] = (1.0 / 2.0) * (
                        self.agents[i].mpc.x.X + self.agents[i - 1].mpc.x_back.X
                    )
                    self.y_back_list[i - 1] += self.rho * (
                        self.agents[i - 1].mpc.x_back.X - self.z_list[i]
                    )
                else:
                    self.z_list[i] = (1.0 / 3.0) * (
                        self.agents[i].mpc.x.X
                        + self.agents[i + 1].mpc.x_front.X
                        + self.agents[i - 1].mpc.x_back.X
                    )
                    self.y_front_list[i + 1] += self.rho * (
                        self.agents[i + 1].mpc.x_front.X - self.z_list[i]
                    )
                    self.y_back_list[i - 1] += self.rho * (
                        self.agents[i - 1].mpc.x_back.X - self.z_list[i]
                    )

                if DEBUG_PLOT:
                    admm_dict["z"][i].append(self.z_list[i])

            # update z and y for local agents
            for i in range(self.n):
                if i == 0:
                    self.agents[i].mpc.set_back_vars(
                        self.y_back_list[i], self.z_list[i + 1]
                    )
                elif i == self.n - 1:
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
            for i in range(self.n):
                tot_resid += sum(
                    np.linalg.norm(
                        admm_dict["x"][i][-1][:, time_step]
                        - admm_dict["z"][i][-1][:, time_step]
                    )
                    for time_step in range(self.N)
                )
                if i != 0:
                    tot_resid += sum(
                        np.linalg.norm(
                            admm_dict["x_back"][i - 1][-1][:, time_step]
                            - admm_dict["z"][i][-1][:, time_step]
                        )
                        for time_step in range(self.N)
                    )
                if i != self.n - 1:
                    tot_resid += sum(
                        np.linalg.norm(
                            admm_dict["x_front"][i + 1][-1][:, time_step]
                            - admm_dict["z"][i][-1][:, time_step]
                        )
                        for time_step in range(self.N)
                    )
            print(f"Total residual = {tot_resid}")
            # centralized solution
            self.cent_mpc.solve_mpc(state)
            time_step = 0  # plot control/state iterations predicted at this time_step
            agent = 2  # plot for which agent
            # plt.plot([admm_dict["u"][agent][i][:, time_step] for i in range(admm_iters)])
            for time_step in range(self.N):
                plt.plot(
                    [
                        admm_dict["u"][agent][i][:, time_step]
                        for i in range(self.admm_iters)
                    ]
                )
                plt.axhline(self.cent_mpc.u.X[agent, time_step], linestyle="--")
                plt.show()

                for a in range(self.n):
                    plt.plot(
                        [
                            admm_dict["x"][a][i][0, time_step]
                            - admm_dict["z"][a][i][0, time_step]
                            for i in range(self.admm_iters)
                        ]
                    )
                plt.title("Residual")
                plt.show()

                plt.plot(
                    [
                        admm_dict["x"][agent][i][0, time_step]
                        for i in range(self.admm_iters)
                    ]
                )
                if agent != 0:
                    plt.plot(
                        [
                            admm_dict["x_back"][agent - 1][i][0, time_step]
                            for i in range(self.admm_iters)
                        ]
                    )
                if agent != self.n - 1:
                    plt.plot(
                        [
                            admm_dict["x_front"][agent + 1][i][0, time_step]
                            for i in range(self.admm_iters)
                        ]
                    )
                plt.axhline(
                    self.cent_mpc.x.X[agent * self.nx_l, time_step], linestyle="--"
                )
                plt.show()

        if u[0].shape[0] > self.nu_l:  # includes gear choices
            # stack the continuous conttrol at the front and the discrete at the back
            return np.vstack(
                (
                    np.vstack([u[i][: self.nu_l, :] for i in range(self.n)]),
                    np.vstack([u[i][self.nu_l :, :] for i in range(self.n)]),
                )
            )
        else:
            return np.vstack(u)

    # here we set the leader cost because it is independent of other vehicles' states
    def on_timestep_end(self, env: Env, episode: int, timestep: int) -> None:
        x_goal = self.leader_x[:, timestep : timestep + self.N + 1]
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
        x_goal = self.leader_x[:, 0 : self.N + 1]
        self.agents[0].mpc.set_x_front(x_goal)
        return super().on_episode_start(env, episode, state)


def simulate(sim: Sim, admm_iters: int = 20, save: bool = False, plot: bool = True):
    n = sim.n  # num cars
    N = sim.N  # controller horizon
    ep_len = sim.ep_len  # length of episode (sim len)
    ts = Params.ts
    masses = sim.masses

    spacing_policy = sim.spacing_policy
    leader_trajectory = sim.leader_trajectory
    leader_x = leader_trajectory.get_leader_trajectory()
    # vehicles
    platoon = Platoon(n, vehicle_type=sim.vehicle_model_type, masses=masses)
    systems = platoon.get_vehicle_system_dicts(ts)

    # env
    env = MonitorEpisodes(
        TimeLimit(
            PlatoonEnv(
                n=n,
                platoon=platoon,
                leader_trajectory=leader_trajectory,
                spacing_policy=spacing_policy,
                start_from_platoon=sim.start_from_platoon,
            ),
            max_episode_steps=ep_len,
        )
    )

    # mpcs
    if sim.vehicle_model_type == "pwa_gear":
        mpc_class = LocalMpcADMM
    elif sim.vehicle_model_type == "pwa_friction":
        mpc_class = LocalMpcGear
    elif sim.vehicle_model_type == "nonlinear":
        raise NotImplementedError()
    else:
        raise ValueError(f"{sim.vehicle_model_type} is not a valid vehicle model type.")
    mpcs = [
        mpc_class(
            N,
            systems[i],
            rho=0.5,
            spacing_policy=spacing_policy,
            is_leader=True if i == 0 else False,
            is_trailer=True if i == n - 1 else False,
        )
        for i in range(n)
    ]
    # agent
    agent = ADMMCoordinator(
        mpcs, admm_iters=admm_iters, rho=0.5, ep_len=ep_len, N=N, leader_x=leader_x, ts=ts
    )

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

    if plot:
        plot_fleet(n, X, U, R, leader_x, violations=env.unwrapped.viol_counter[0])

    if save:
        with open(
            f"admm_{admm_iters}_{sim.id}" + ".pkl",
            "wb",
        ) as file:
            pickle.dump(X, file)
            pickle.dump(U, file)
            pickle.dump(R, file)
            pickle.dump(agent.solve_times, file)
            pickle.dump(agent.node_counts, file)
            pickle.dump(env.unwrapped.viol_counter[0], file)
            pickle.dump(leader_x, file)


if __name__ == "__main__":
    simulate(Sim())
