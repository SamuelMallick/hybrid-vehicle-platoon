import pickle

import casadi as cs
import numpy as np
from csnlp import Nlp
from dmpcpwa.agents.g_admm_coordinator import GAdmmCoordinator
from dmpcpwa.mpc.mpc_switching import MpcSwitching
from dmpcrl.core.admm import g_map
from dmpcrl.mpc.mpc_admm import MpcAdmm
from gymnasium.wrappers import TimeLimit
from mpcrl.wrappers.envs import MonitorEpisodes

from env import PlatoonEnv
from misc.common_controller_params import Params, Sim
from misc.spacing_policy import ConstantSpacingPolicy, SpacingPolicy
from models import Platoon, Vehicle
from plot_fleet import plot_fleet

np.random.seed(2)


class LocalMpc(MpcSwitching):

    Q_x = Params.Q_x
    Q_u = Params.Q_u
    Q_du = Params.Q_du
    w = Params.w
    a_acc = Params.a_acc
    a_dec = Params.a_dec
    ts = Params.ts
    d_safe = Params.d_safe

    rho = 0.5

    def __init__(
        self,
        N: int,
        pwa_system: dict,
        num_neighbours: int,
        my_index: int,
        spacing_policy: SpacingPolicy = ConstantSpacingPolicy(50),
        leader: bool = False,
    ) -> None:
        """Instantiate inner switching MPC for admm for car fleet. If leader is true the cost uses the reference traj
        My index is used to pick out own state from the grouped coupling states.
        It should be passed in via the mapping G (G[i].index(i))"""
        self.horizon=N  # needed for Nlp object
        nlp = Nlp[cs.SX]()
        super().__init__(nlp, N)
        nx_l = Vehicle.nx_l
        nu_l = Vehicle.nu_l
        self.nx_l = nx_l
        self.nu_l = nu_l

        x, x_c = self.augmented_state(num_neighbours, my_index, size=nx_l)

        u, _ = self.action(
            "u",
            nu_l,
        )

        s, _, _ = self.variable(
            "s",
            (1, N + 1),
            lb=0,
        )  # slack var for safe distance constraint

        x_c_list = (
            []
        )  # store the bits of x that are couplings in a list for ease of access
        for i in range(num_neighbours):
            x_c_list.append(x_c[nx_l * i : nx_l * (i + 1), :])

        r = pwa_system["T"][0].shape[
            0
        ]  # number of conditions when constraining a region
        self.set_dynamics(nx_l, nu_l, r, x, u, x_c_list)

        # normal constraints
        for k in range(N + 1):
            self.constraint(
                f"state_{k}", pwa_system["D"] @ x[:, [k]], "<=", pwa_system["E"]
            )
        for k in range(N):
            self.constraint(
                f"control_{k}", pwa_system["F"] @ u[:, [k]], "<=", pwa_system["G"]
            )

        for k in range(N):
            # acceleration limits
            self.constraint(
                f"acc_{k}", x[1, [k + 1]] - x[1, [k]], "<=", self.a_acc * self.ts
            )
            self.constraint(
                f"de_acc_{k}", x[1, [k + 1]] - x[1, [k]], ">=", self.a_dec * self.ts
            )

            # safety constraints - if leader they are done later with parameters
            if not leader:
                self.constraint(
                    f"safety_{k}",
                    x[0, [k]],
                    "<=",
                    x_c[0, [k]] - self.d_safe + s[:, [k]],
                )
        if not leader:
            self.constraint(
                f"safety_{N}", x[0, [N]], "<=", x_c[0, [N]] - self.d_safe + s[:, [N]]
            )

        # objective
        if leader:
            self.leader_traj = []
            for k in range(N + 1):
                temp = self.parameter(f"x_ref_{k}", (nx_l, 1))
                self.leader_traj.append(temp)
                self.fixed_pars_init[f"x_ref_{k}"] = np.zeros((nx_l, 1))
            self.set_local_cost(
                sum(
                    (x[:, [k]] - self.leader_traj[k]).T
                    @ self.Q_x
                    @ (x[:, [k]] - self.leader_traj[k])
                    + u[:, [k]].T @ self.Q_u @ u[:, [k]]
                    for k in range(N)
                )
                + sum(
                    (u[:, [k + 1]] - u[:, [k]]).T
                    @ self.Q_du
                    @ (u[:, [k + 1]] - u[:, [k]])
                    for k in range(N - 1)
                )
                + (x[:, [N]] - self.leader_traj[N]).T
                @ self.Q_x
                @ (x[:, [N]] - self.leader_traj[N])
            )
        else:
            # following the agent ahead - therefore the index of the local state copy to track
            # is always the FIRST one in the local copies x_c
            self.set_local_cost(
                sum(
                    (x[:, [k]] - x_c[0:nx_l, [k]] - spacing_policy.spacing(x[:, [k]])).T
                    @ self.Q_x
                    @ (x[:, [k]] - x_c[0:nx_l, [k]] - spacing_policy.spacing(x[:, [k]]))
                    + u[:, [k]].T @ self.Q_u @ u[:, [k]]
                    + self.w * s[:, [k]]
                    for k in range(N)
                )
                + sum(
                    (u[:, [k + 1]] - u[:, [k]]).T
                    @ self.Q_du
                    @ (u[:, [k + 1]] - u[:, [k]])
                    for k in range(N - 1)
                )
                + (x[:, [N]] - x_c[0:nx_l, [N]] - spacing_policy.spacing(x[:, [N]])).T
                @ self.Q_x
                @ (x[:, [N]] - x_c[0:nx_l, [N]] - spacing_policy.spacing(x[:, [N]]))
                + self.w * s[:, [N]]
            )

        # solver

        solver = "qpoases"  # "qpoases"
        if solver == "ipopt":
            opts = {
                "expand": True,
                "show_eval_warnings": True,
                "warn_initial_bounds": True,
                "print_time": False,
                "record_time": True,
                "bound_consistency": True,
                "calc_lam_x": True,
                "calc_lam_p": False,
                # "jit": True,
                # "jit_cleanup": True,
                "ipopt": {
                    # "linear_solver": "ma97",
                    # "linear_system_scaling": "mc19",
                    # "nlp_scaling_method": "equilibration-based",
                    "max_iter": 500,
                    "sb": "yes",
                    "print_level": 0,
                },
            }
        elif solver == "qrqp":
            opts = {
                "expand": True,
                "print_time": False,
                "record_time": True,
                "error_on_fail": False,
                "print_info": False,
                "print_iter": False,
                "print_header": False,
                "max_iter": 2000,
            }
        elif solver == "qpoases":
            opts = {
                "print_time": False,
                "record_time": True,
                "error_on_fail": True,
                "printLevel": "none",
            }
        else:
            raise RuntimeError("No solver type defined.")

        self.init_solver(opts, solver=solver)


class TrackingGAdmmCoordinator(GAdmmCoordinator):
    best_warm_starts = []
    solve_times = []

    def __init__(
        self,
        N: int,
        ep_len: int,
        leader_x: np.ndarray,
        local_mpcs: list[MpcAdmm],
        local_fixed_parameters: list[dict],
        systems: list[dict],
        vehicles: list[Vehicle],
        G: list[list[int]],
        Adj: np.ndarray,
        rho: float,
        debug_plot: bool = False,
        admm_iters: int = 50
    ) -> None:
        super().__init__(
            local_mpcs,
            local_fixed_parameters,
            systems,
            G,
            Adj,
            rho,
            debug_plot,
            admm_iters,
        )
        self.N = N
        self.ep_len = ep_len
        self.leader_x = leader_x
        self.vehicles = vehicles

    def on_timestep_end(self, env, episode: int, timestep: int) -> None:
        # time step starts from 1, so this will set the cost accurately for the next time-step
        self.set_leader_traj(self.leader_x[:, timestep : (timestep + self.N + 1)])
        return super().on_timestep_end(env, episode, timestep)

    def on_episode_start(self, env, episode: int, state) -> None:
        self.set_leader_traj(self.leader_x[:, 0 : self.N + 1])
        return super().on_episode_start(env, episode, state)

    def set_leader_traj(self, leader_traj):
        for k in range(self.N + 1):  # we assume first agent is leader!
            self.agents[0].fixed_parameters[f"x_ref_{k}"] = leader_traj[:, [k]]

    def g_admm_control(self, state, warm_start=None):
        # set warm start for fleet: constant velocity
        warm_start = [
            [
                self.vehicles[i].get_u_for_constant_vel(state[2 * i + 1, 0])
                * np.ones((self.nu_l, self.N))
                for i in range(self.n)
            ]
        ]

        # shifted previous solution as warm start
        if self.prev_sol is not None:
            warm_start.append(
                [
                    np.hstack((self.prev_sol[i][:, 1:], self.prev_sol[i][:, [-1]]))
                    for i in range(self.n)
                ]
            )

        best_cost = float("inf")
        best_control = [np.zeros((self.nu_l, self.N)) for i in range(self.n)]
        counter = 0
        self.best_warm_starts.append(counter)

        temp_solve_times = []
        for u in warm_start:
            counter += 1
            u_opt, sol_list, error_flag, infeas_flag = super().g_admm_control(
                state, warm_start=u
            )
            if not error_flag and not infeas_flag:
                cost = sum(sol_list[i].f for i in range(self.n))
                temp_solve_times.append(self.prev_sol_time)
            else:
                cost = float("inf")
            if cost < best_cost:
                best_cost = cost
                best_control = u_opt
                self.best_warm_starts[-1] = counter

        if best_cost == float("inf"):
            self.solve_times.append(0.0)
            raise RuntimeError("No solution found for any of the warm starts")
            # return cs.DM(warm_start[0]), sol_list, True, True
        else:
            self.solve_times.append(max(temp_solve_times))
            return cs.DM(best_control), None, None, None


def simulate(
    sim: Sim,
    save: bool = False,
    plot: bool = True,
    seed: int = 1,
):
    n = sim.n  # num cars
    N = sim.N  # controller horizon
    ep_len = sim.ep_len  # length of episode (sim len)
    ts = Params.ts
    masses = sim.masses

    nx_l = Vehicle.nx_l
    Vehicle.nu_l

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
    # no state coupling here so all zeros
    Ac = np.zeros((nx_l, nx_l))
    for i in range(n):
        # add the coupling part of the system
        Ac_i = []
        for j in range(n):
            if Adj[i, j] == 1:
                Ac_i.append(Ac)
        systems[i]["Ac"] = []
        for j in range(
            len(systems[i]["S"])
        ):  # duplicate it for each PWA region, as for this PWA system the coupling matrices do not change
            systems[i]["Ac"] = systems[i]["Ac"] + [Ac_i]

    if sim.vehicle_model_type == "pwa_gear":
        pass
    elif sim.vehicle_model_type == "pwa_friction":
        raise NotImplementedError()
    elif sim.vehicle_model_type == "nonlinear":
        raise NotImplementedError()
    else:
        raise ValueError(f"{sim.vehicle_model_type} is not a valid vehicle model type.")
    # distributed mpcs and params
    local_mpcs: list[LocalMpc] = []
    local_fixed_dist_parameters: list[dict] = []
    for i in range(n):
        local_mpcs.append(
            LocalMpc(
                N=N,
                pwa_system=systems[i],
                spacing_policy=spacing_policy,
                num_neighbours=len(G_map[i]) - 1,
                my_index=G_map[i].index(i),
                leader=True if i == 0 else False,
            )
        )
        local_fixed_dist_parameters.append(local_mpcs[i].fixed_pars_init)

    # agent
    agent = TrackingGAdmmCoordinator(
        N=N,
        ep_len=ep_len,
        leader_x=leader_x,
        local_mpcs=local_mpcs,
        local_fixed_parameters=local_fixed_dist_parameters,
        systems=systems,
        vehicles=platoon.get_vehicles(),
        G=G_map,
        Adj=Adj,
        rho=LocalMpc.rho,
        debug_plot=False,
        admm_iters=100,
    )

    agent.evaluate(env=env, episodes=1, seed=seed)

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
            f"switching_admm_{sim.id}_seed_{seed}" + ".pkl",
            "wb",
        ) as file:
            pickle.dump(X, file)
            pickle.dump(U, file)
            pickle.dump(R, file)
            pickle.dump(agent.solve_times, file)
            pickle.dump(0, file)    # empty dump where the other approaches put node counts
            pickle.dump(env.unwrapped.viol_counter[0], file)
            pickle.dump(leader_x, file)


if __name__ == "__main__":
    simulate(Sim(), save=True, seed=2)
