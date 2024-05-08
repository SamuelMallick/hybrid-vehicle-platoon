import pickle

import gurobipy as gp
import numpy as np
from dmpcpwa.agents.mld_agent import MldAgent
from dmpcpwa.mpc.mpc_mld import MpcMld
from gymnasium.wrappers import TimeLimit
from mpcrl.wrappers.envs import MonitorEpisodes

from env import PlatoonEnv
from misc.common_controller_params import Params, Sim
from misc.spacing_policy import ConstantSpacingPolicy, SpacingPolicy
from models import Platoon, Vehicle
from mpcs.mpc_gear import MpcGear
from plot_fleet import plot_fleet

np.random.seed(2)


class LocalMpcMld(MpcMld):
    """A local decentralized MPC for a single vehicle in the platoon."""

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
        spacing_policy: SpacingPolicy = ConstantSpacingPolicy(50),
        quadratic_cost: bool = True,
        is_front: bool = False,
        is_leader: bool = False,
        is_trailer: bool = False,
        thread_limit: int | None = None,
        accel_cnstr_tightening: float = 0.0,
        real_vehicle_as_reference: bool = False,
    ) -> None:
        super().__init__(
            pwa_system, N, thread_limit=thread_limit, constrain_first_state=False
        )
        self.N = N
        self.setup_cost_and_constraints(
            self.u,
            spacing_policy,
            quadratic_cost,
            is_front,
            is_leader,
            is_trailer,
            accel_cnstr_tightening,
            real_vehicle_as_reference,
        )

    def setup_cost_and_constraints(
        self,
        u,
        spacing_policy: SpacingPolicy = ConstantSpacingPolicy(50),
        quadratic_cost: bool = True,
        is_front: bool = False,
        is_leader=False,
        is_trailer=False,
        accel_cnstr_tightening: float = 0.0,
        real_vehicle_as_reference: bool = False,
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
        if is_leader:
            self.leader_x = self.mpc_model.addMVar(
                (nx_l, self.N + 1), lb=-1e6, ub=1e6, name="leader_x"
            )
        # slack vars for constraints with prec and succ vehicles
        self.s_front = self.mpc_model.addMVar(
            (self.N + 1), lb=0, ub=float("inf"), name="s_front"
        )
        self.s_back = self.mpc_model.addMVar(
            (self.N + 1), lb=0, ub=float("inf"), name="s_back"
        )

        # setting these bounds to zero removes the slack var, as leader and trailer
        # dont have cars in front or behind respectively
        if is_front:
            self.s_front.ub = 0
        if is_trailer:
            self.s_back.ub = 0

        # cost func
        cost = 0
        # tracking cost
        if not is_front and not is_leader:
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
        if not is_trailer and not is_leader:
            cost += sum(
                [
                    self.cost_func(
                        self.x_back[:, [k]]
                        - self.x[:, [k]]
                        - spacing_policy.spacing(self.x_back[:, [k]]),
                        self.Q_x,
                    )
                    for k in range(self.N + 1)
                ]
            )
        if is_leader:
            if not real_vehicle_as_reference:
                cost += sum(
                    [
                        self.cost_func(self.x[:, [k]] - self.leader_x[:, [k]], self.Q_x)
                        for k in range(self.N + 1)
                    ]
                )
            else:
                cost += sum(
                    [
                        self.cost_func(
                            self.x[:, [k]]
                            - self.leader_x[:, [k]]
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
        self.mpc_model.setObjective(cost, gp.GRB.MINIMIZE)

        # accel constraints
        self.mpc_model.addConstrs(
            (
                self.a_dec * self.ts
                <= self.x[1, [k + 1]] - self.x[1, [k]] - k * accel_cnstr_tightening
                for k in range(self.N)
            ),
            name="dec",
        )
        self.mpc_model.addConstrs(
            (
                self.x[1, [k + 1]] - self.x[1, [k]]
                <= self.a_acc * self.ts - k * accel_cnstr_tightening
                for k in range(self.N)
            ),
            name="acc",
        )

        # safe distance constraints
        if not is_front:
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

    def set_leader_x(self, leader_x):
        for k in range(self.N + 1):
            self.leader_x[:, [k]].lb = leader_x[:, [k]]
            self.leader_x[:, [k]].ub = leader_x[:, [k]]

    def set_x_front(self, x_front):
        for k in range(self.N + 1):
            self.x_front[:, [k]].lb = x_front[:, [k]]
            self.x_front[:, [k]].ub = x_front[:, [k]]

    def set_x_back(self, x_back):
        for k in range(self.N + 1):
            self.x_back[:, [k]].lb = x_back[:, [k]]
            self.x_back[:, [k]].ub = x_back[:, [k]]


class LocalMpcGear(LocalMpcMld, MpcGear):
    def __init__(
        self,
        N: int,
        pwa_system: dict,
        spacing_policy: SpacingPolicy = ConstantSpacingPolicy(50),
        quadratic_cost: bool = True,
        is_front: bool = False,
        is_leader: bool = False,
        is_trailer: bool = False,
        thread_limit: int | None = None,
        accel_cnstr_tightening: float = 0.0,
        real_vehicle_as_reference: bool = False,
    ) -> None:
        MpcGear.__init__(
            self, pwa_system, N, thread_limit=thread_limit, constrain_first_state=False
        )
        self.setup_gears(N, pwa_system["F"], pwa_system["G"])
        self.setup_cost_and_constraints(
            self.u_g,
            spacing_policy,
            quadratic_cost,
            is_front,
            is_leader,
            is_trailer,
            accel_cnstr_tightening,
            real_vehicle_as_reference,
        )


class TrackingDecentMldCoordinator(MldAgent):
    def __init__(
        self,
        local_mpcs: list[MpcMld],
        ep_len: int,
        N: int,
        leader_x: np.ndarray,
        ts: float,
        leader_index: int = 0,
        velocity_estimator: bool = False,
    ) -> None:
        super().__init__(local_mpcs[0])  # just to handle agent inititalisation

        self.n = len(local_mpcs)
        self.ep_len = ep_len
        self.ts = ts
        self.N = N
        self.leader_x = leader_x
        self.leader_index = leader_index
        self.velocity_estimator = velocity_estimator
        self.nx_l = Vehicle.nx_l
        self.nu_l = Vehicle.nu_l
        self.agents: list[MldAgent] = []
        for i in range(self.n):
            self.agents.append(MldAgent(local_mpcs[i]))

        self.solve_times = np.zeros((ep_len, 1))
        self.node_counts = np.zeros((ep_len, 1))

    def get_control(self, state: np.ndarray):
        x_l = np.split(state, self.n, axis=0)
        u = [self.agents[i].get_control(x_l[i]) for i in range(self.n)]
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

    # current state of car to be tracked is observed and propogated forward
    # to be the prediction
    def on_timestep_end(self, env: PlatoonEnv, episode: int, timestep: int) -> None:
        x_goal = self.leader_x[:, timestep : timestep + self.N + 1]
        self.agents[self.leader_index].mpc.set_leader_x(x_goal)
        self.observe_states(env, timestep)

        # take the maximum solve time and maximum node count, as all agents are assumed to
        # have operated in parallel
        agent_solve_times = [self.agents[i].run_time for i in range(self.n)]
        agent_node_counts = [self.agents[i].node_count for i in range(self.n)]
        self.solve_times[env.step_counter - 1, :] = max(agent_solve_times)
        self.node_counts[env.step_counter - 1, :] = max(agent_node_counts)
        return super().on_timestep_end(env, episode, timestep)

    def on_episode_start(self, env: PlatoonEnv, episode: int, state) -> None:
        x_goal = self.leader_x[:, 0 : self.N + 1]
        self.agents[self.leader_index].mpc.set_leader_x(x_goal)
        self.observe_states(env, timestep=0)
        return super().on_episode_start(env, episode, state)

    def observe_states(self, env: PlatoonEnv, timestep):
        if self.velocity_estimator:
            x_p = env.get_previous_state()
        for i in range(self.n):
            if i == 0:  # lead car
                if self.velocity_estimator:
                    x_pred_back = self.extrapolate_position_two_point_estimator(
                        env.x[self.nx_l * (i + 1)],
                        env.x[self.nx_l * (i + 1) + 1],
                        x_p[self.nx_l * (i + 1) + 1],
                    )
                else:
                    x_pred_back = self.extrapolate_position_constant_vel(
                        env.x[self.nx_l * (i + 1)], env.x[self.nx_l * (i + 1) + 1]
                    )
                self.agents[i].mpc.set_x_back(x_pred_back)
            elif i == self.n - 1:  # last car
                if self.velocity_estimator:
                    x_pred_front = self.extrapolate_position_two_point_estimator(
                        env.x[self.nx_l * (i - 1)],
                        env.x[self.nx_l * (i - 1) + 1],
                        x_p[self.nx_l * (i - 1) + 1],
                    )
                else:
                    x_pred_front = self.extrapolate_position_constant_vel(
                        env.x[self.nx_l * (i - 1)], env.x[self.nx_l * (i - 1) + 1]
                    )
                self.agents[i].mpc.set_x_front(x_pred_front)
            else:
                if self.velocity_estimator:
                    x_pred_front = self.extrapolate_position_two_point_estimator(
                        env.x[self.nx_l * (i - 1)],
                        env.x[self.nx_l * (i - 1) + 1],
                        x_p[self.nx_l * (i - 1) + 1],
                    )
                    x_pred_back = self.extrapolate_position_two_point_estimator(
                        env.x[self.nx_l * (i + 1)],
                        env.x[self.nx_l * (i + 1) + 1],
                        x_p[self.nx_l * (i + 1) + 1],
                    )
                else:
                    x_pred_front = self.extrapolate_position_constant_vel(
                        env.x[self.nx_l * (i - 1)], env.x[self.nx_l * (i - 1) + 1]
                    )
                    x_pred_back = self.extrapolate_position_constant_vel(
                        env.x[self.nx_l * (i + 1)], env.x[self.nx_l * (i + 1) + 1]
                    )
                self.agents[i].mpc.set_x_front(x_pred_front)
                self.agents[i].mpc.set_x_back(x_pred_back)

    def extrapolate_position_constant_vel(self, initial_pos: float, initial_vel: float):
        x_pred = np.zeros((self.nx_l, self.N + 1))
        x_pred[0, [0]] = initial_pos
        x_pred[1, [0]] = initial_vel
        for k in range(self.N):
            x_pred[0, [k + 1]] = x_pred[0, [k]] + self.ts * x_pred[1, [k]]
            x_pred[1, [k + 1]] = x_pred[1, [k]]
        return x_pred

    def extrapolate_position_two_point_estimator(
        self, initial_pos: float, initial_vel: float, previous_vel: float
    ):
        x_pred = np.zeros((self.nx_l, self.N + 1))
        x_pred[0, [0]] = initial_pos
        x_pred[1, [0]] = initial_vel
        dv = initial_vel - previous_vel
        for k in range(self.N):
            x_pred[0, [k + 1]] = x_pred[0, [k]] + self.ts * x_pred[1, [k]]
            x_pred[1, [k + 1]] = x_pred[1, [k]] + dv
        return x_pred


def simulate(
    sim: Sim,
    save: bool = False,
    plot: bool = True,
    seed: int = 2,
    thread_limit: int | None = None,
    velocity_estimator: bool = False,
    leader_index: int = 0,
):
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
                real_vehicle_as_reference=sim.real_vehicle_as_reference,
                ep_len=sim.ep_len,
                leader_index=leader_index,
            ),
            max_episode_steps=ep_len,
        )
    )

    # mpcs
    if sim.vehicle_model_type == "pwa_gear":
        mpc_class = LocalMpcMld
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
            spacing_policy,
            is_front=True if i == 0 else False,
            is_leader=True if i == leader_index else False,
            is_trailer=True if i == n - 1 else False,
            thread_limit=thread_limit,
            real_vehicle_as_reference=sim.real_vehicle_as_reference,
        )
        for i in range(n)
    ]
    # agent
    agent = TrackingDecentMldCoordinator(
        mpcs,
        ep_len=ep_len,
        N=N,
        leader_x=leader_x,
        ts=ts,
        velocity_estimator=velocity_estimator,
        leader_index=leader_index,
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
            f"decent_vest_{velocity_estimator}_{sim.id}_seed_{seed}" + ".pkl",
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
    simulate(Sim(), save=False, seed=1, velocity_estimator=False, leader_index=0)
