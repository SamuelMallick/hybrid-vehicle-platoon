import pickle

import gurobipy as gp
import numpy as np
from dmpcpwa.agents.mld_agent import MldAgent
from dmpcpwa.mpc.mpc_mld_cent_decup import MpcMldCentDecup
from gymnasium import Env
from gymnasium.wrappers import TimeLimit
from mpcrl.wrappers.envs import MonitorEpisodes
from scipy.linalg import block_diag

from env import PlatoonEnv
from misc.common_controller_params import Params, Sim
from misc.spacing_policy import ConstantSpacingPolicy, SpacingPolicy
from models import Platoon, Vehicle
from mpcs.mpc_gear import MpcGear, MpcNonlinearGear
from utils.plot_fleet import plot_fleet

np.random.seed(2)

DEBUG = False

threshold = 10  # cost improvement must be more than this to consider communication


class LocalMpc(MpcMldCentDecup):
    """Mpc for a vehicle, solving for states of vehicle in front and behind. Local state has car
    is organised with x = [x_front, x_me, x_back]."""

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
        systems: list[dict],
        num_vehicles_in_front: int,
        num_vehicles_behind: int,
        spacing_policy: SpacingPolicy = ConstantSpacingPolicy(50),
        rel_leader_index: int | None = None,
        quadratic_cost: bool = True,
        thread_limit: int | None = None,
        accel_cnstr_tightening: float = 0.0,
    ) -> None:
        """Initialize the local MPC. This MPC optimizes also considering neighboring vehicles.
        The number of neighboring vehicles is passed in through num_vehicles_in_front/num_vehicles_behind.
        """
        self.n = len(systems)
        super().__init__(
            systems,
            self.n,
            N,
            thread_limit=thread_limit,
            constrain_first_state=False,
        )
        self.setup_cost_and_constraints(
            self.u,
            num_vehicles_in_front,
            num_vehicles_behind,
            spacing_policy,
            rel_leader_index,
            quadratic_cost,
            accel_cnstr_tightening,
        )

    def setup_cost_and_constraints(
        self,
        u,
        num_vehicles_in_front: int,
        num_vehicles_behind: int,
        spacing_policy: SpacingPolicy = ConstantSpacingPolicy(50),
        rel_leader_index: int | None = None,
        quadratic_cost: bool = True,
        accel_cnstr_tightening: float = 0.0,
    ):
        """Set up  cost and constraints for vehicle. Penalises the u passed in."""
        if quadratic_cost:
            self.cost_func = self.min_2_norm
        else:
            self.cost_func = self.min_1_norm

        nx_l = Vehicle.nx_l
        self.nu_l = Vehicle.nu_l

        # break up state and control into local and that of other vehicles
        if num_vehicles_in_front > 0:
            x_f1 = self.x[0:nx_l, :]  # state of vehicle directly in front
            x_me = self.x[nx_l : 2 * nx_l, :]  # local state
            if num_vehicles_behind > 0:
                x_b1 = self.x[
                    2 * nx_l : 3 * nx_l, :
                ]  # state of vehicle directly behind
        else:
            x_me = self.x[0:nx_l, :]  # local state
            if num_vehicles_behind > 0:
                x_b1 = self.x[nx_l : 2 * nx_l, :]  # state of vehicle directly behind

        x_l = [self.x[i * nx_l : (i + 1) * nx_l, :] for i in range(self.n)]
        u_l = [u[i * self.nu_l : (i + 1) * self.nu_l, :] for i in range(self.n)]

        if rel_leader_index is not None:
            self.leader_x = self.mpc_model.addMVar(
                (nx_l, self.N + 1), lb=0, ub=0, name="leader_x"
            )

        # slack vars for constraints with vehicles in front and behind
        self.s_f1 = self.mpc_model.addMVar(
            (self.N + 1), lb=0, ub=float("inf"), name="s_f1"
        )
        self.s_b1 = self.mpc_model.addMVar(
            (self.N + 1), lb=0, ub=float("inf"), name="s_b1"
        )
        self.s_f2 = self.mpc_model.addMVar(
            (self.N + 1), lb=0, ub=float("inf"), name="s_f2"
        )
        self.s_b2 = self.mpc_model.addMVar(
            (self.N + 1), lb=0, ub=float("inf"), name="s_b2"
        )
        # set these slacks to 0 if no associated vehilces. Also create state copies for vehicles 1 neighbor away
        if num_vehicles_in_front < 2:
            self.s_f2.ub = 0
            if num_vehicles_in_front < 1:
                self.s_f1.ub = 0
        else:
            self.x_f2 = self.mpc_model.addMVar(
                (nx_l, self.N + 1), lb=-1e6, ub=1e6, name="x_f2"
            )
        if num_vehicles_behind < 2:
            self.s_b2.ub = 0
            if num_vehicles_behind < 1:
                self.s_b1.ub = 0
        else:
            self.x_b2 = self.mpc_model.addMVar(
                (nx_l, self.N + 1), lb=-1e6, ub=1e6, name="x_b2"
            )

        # cost func
        cost = 0
        # leader tracking cost
        if rel_leader_index is not None:
            if rel_leader_index == -1:
                cost += sum(
                    self.cost_func(x_b1[:, [k]] - self.leader_x[:, [k]], self.Q_x)
                    for k in range(self.N + 1)
                )
            elif rel_leader_index == 0:
                cost += sum(
                    self.cost_func(x_me[:, [k]] - self.leader_x[:, [k]], self.Q_x)
                    for k in range(self.N + 1)
                )
            elif rel_leader_index == 1:
                cost += sum(
                    self.cost_func(x_f1[:, [k]] - self.leader_x[:, [k]], self.Q_x)
                    for k in range(self.N + 1)
                )
            else:
                raise ValueError(
                    f"rel leader index must be -1, 0, or 1. Got {rel_leader_index}."
                )

        # tracking cost for vehicles ahead
        if num_vehicles_in_front > 0:
            cost += sum(
                self.cost_func(
                    x_me[:, [k]] - x_f1[:, [k]] - spacing_policy.spacing(x_me[:, [k]]),
                    self.Q_x,
                )
                for k in range(self.N + 1)
            )
            if num_vehicles_in_front > 1:
                cost += sum(
                    self.cost_func(
                        x_f1[:, [k]]
                        - self.x_f2[:, [k]]
                        - spacing_policy.spacing(x_f1[:, [k]]),
                        self.Q_x,
                    )
                    for k in range(self.N + 1)
                )
        # trcking cost for vehicles behind
        if num_vehicles_behind > 0:
            cost += sum(
                [
                    self.cost_func(
                        x_b1[:, [k]]
                        - x_me[:, [k]]
                        - spacing_policy.spacing(x_b1[:, [k]]),
                        self.Q_x,
                    )
                    for k in range(self.N + 1)
                ]
            )
            if num_vehicles_behind > 1:
                cost += sum(
                    [
                        self.cost_func(
                            self.x_b2[:, [k]]
                            - x_b1[:, [k]]
                            - spacing_policy.spacing(self.x_b2[:, [k]]),
                            self.Q_x,
                        )
                        for k in range(self.N + 1)
                    ]
                )
        # control effort cost
        cost += sum(
            [
                self.cost_func(u_l[i][:, [k]], self.Q_u)
                for k in range(self.N)
                for i in range(self.n)
            ]
        )
        # control variation cost
        cost += sum(
            [
                self.cost_func(u_l[i][:, [k + 1]] - u_l[i][:, [k]], self.Q_du)
                for k in range(self.N - 1)
                for i in range(self.n)
            ]
        )
        # slack variable cost
        cost += sum(
            [
                self.w * self.s_f1[k]
                + self.w * self.s_f2[k]
                + self.w * self.s_b1[k]
                + self.w * self.s_b2[k]
                for k in range(self.N + 1)
            ]
        )
        self.mpc_model.setObjective(cost, gp.GRB.MINIMIZE)

        # accel constraints
        self.mpc_model.addConstrs(
            (
                self.a_dec * self.ts
                <= x_l[i][1, k + 1] - x_l[i][1, k] - k * accel_cnstr_tightening
                for i in range(self.n)
                for k in range(self.N)
            ),
            name="dec",
        )
        self.mpc_model.addConstrs(
            (
                x_l[i][1, k + 1] - x_l[i][1, k]
                <= self.a_acc * self.ts - k * accel_cnstr_tightening
                for i in range(self.n)
                for k in range(self.N)
            ),
            name="acc",
        )

        # safe distance constraints
        if num_vehicles_in_front > 0:
            self.mpc_model.addConstrs(
                (
                    x_me[0, k] <= x_f1[0, k] - self.d_safe + self.s_f1[k]
                    for k in range(self.N + 1)
                ),
                name="safe_f1",
            )
            if num_vehicles_in_front > 1:
                self.mpc_model.addConstrs(
                    (
                        x_f1[0, k] <= self.x_f2[0, k] - self.d_safe + self.s_f2[k]
                        for k in range(self.N + 1)
                    ),
                    name="safe_f2",
                )
        if num_vehicles_behind > 0:
            self.mpc_model.addConstrs(
                (
                    x_b1[0, k] <= x_me[0, k] - self.d_safe + self.s_b1[k]
                    for k in range(self.N + 1)
                ),
                name="safe_b1",
            )
            if num_vehicles_behind > 1:
                self.mpc_model.addConstrs(
                    (
                        self.x_b2[0, k] <= x_b1[0, k] - self.d_safe + self.s_b2[k]
                        for k in range(self.N + 1)
                    ),
                    name="safe_b2",
                )

    def set_leader_x(self, leader_x):
        for k in range(self.N + 1):
            self.leader_x[:, [k]].lb = leader_x[:, [k]]
            self.leader_x[:, [k]].ub = leader_x[:, [k]]

    def set_x_f2(self, x_f2):
        for k in range(self.N + 1):
            self.x_f2[:, [k]].lb = x_f2[:, [k]]
            self.x_f2[:, [k]].ub = x_f2[:, [k]]

    def set_x_b2(self, x_b2):
        for k in range(self.N + 1):
            self.x_b2[:, [k]].lb = x_b2[:, [k]]
            self.x_b2[:, [k]].ub = x_b2[:, [k]]

    def eval_cost(self, x: np.ndarray, u: np.ndarray, discrete_gears: bool = False):
        # set the bounds of the vars in the model to fix the vals
        for k in range(
            self.N
        ):  # we dont constain the N+1th state, as it is defined by shifted control
            self.x[:, [k]].ub = x[:, [k]]
            self.x[:, [k]].lb = x[:, [k]]
        if discrete_gears:
            self.u_g.ub = u
            self.u_g.lb = u
        else:
            self.u.ub = u
            self.u.lb = u
        self.IC.RHS = x[:, [0]]
        self.mpc_model.optimize()
        if self.mpc_model.Status == 2:  # check for successful solve
            cost = self.mpc_model.objVal
        else:
            cost = float("inf")  # infinite cost if infeasible
        return cost

    def solve_mpc(self, state, raises: bool = False):
        # the solve method is overridden so that the bounds on the vars are set back to normal before solving.
        self.x.ub = float("inf")
        self.x.lb = -float("inf")
        try:  # if u_g is not defined it means this mpc is not using discrete gears
            self.u_g.ub = float("inf")
            self.u_g.lb = -float("inf")
        except:
            pass
        self.u.ub = float("inf")
        self.u.lb = -float("inf")
        return super().solve_mpc(state, raises=raises)


class LocalMpcGear(LocalMpc, MpcMldCentDecup, MpcGear):
    def __init__(
        self,
        N: int,
        systems: list[dict],
        num_vehicles_in_front: int,
        num_vehicles_behind: int,
        spacing_policy: SpacingPolicy = ConstantSpacingPolicy(50),
        rel_leader_index: int | None = None,
        quadratic_cost: bool = True,
        thread_limit: int | None = None,
        accel_cnstr_tightening: float = 0.0,
    ) -> None:
        self.n = len(systems)
        MpcMldCentDecup.__init__(
            self,
            systems,
            self.n,
            N,
            thread_limit=thread_limit,
            constrain_first_state=False,
        )
        F = block_diag(*([systems[i]["F"] for i in range(self.n)]))
        G = np.vstack([systems[i]["G"] for i in range(self.n)])
        self.setup_gears(N, F, G)
        self.setup_cost_and_constraints(
            self.u_g,
            num_vehicles_in_front,
            num_vehicles_behind,
            spacing_policy,
            rel_leader_index,
            quadratic_cost,
            accel_cnstr_tightening,
        )


class LocalMpcNonlinearGear(LocalMpc, MpcMldCentDecup, MpcNonlinearGear):
    def __init__(
        self,
        N: int,
        systems: list[dict],
        num_vehicles_in_front: int,
        num_vehicles_behind: int,
        spacing_policy: SpacingPolicy = ConstantSpacingPolicy(50),
        rel_leader_index: int | None = None,
        quadratic_cost: bool = True,
        thread_limit: int | None = None,
        accel_cnstr_tightening: float = 0.0,
    ) -> None:
        self.n = len(systems)
        MpcNonlinearGear.__init__(
            self,
            systems,
            N,
            thread_limit=thread_limit,
            constrain_first_state=False,
        )
        F = block_diag(*([systems[i]["F"] for i in range(self.n)]))
        G = np.vstack([systems[i]["G"] for i in range(self.n)])
        self.setup_gears(N, F, G)
        self.setup_cost_and_constraints(
            self.u_g,
            num_vehicles_in_front,
            num_vehicles_behind,
            spacing_policy,
            rel_leader_index,
            quadratic_cost,
            accel_cnstr_tightening,
        )


class TrackingEventBasedCoordinator(MldAgent):
    def __init__(
        self,
        local_mpcs: list[LocalMpc],
        vehicles: list[Vehicle],
        ep_len: int,
        N: int,
        leader_x: np.ndarray,
        discrete_gears: bool,
        ts: float,
        event_iters: int = 4,
        leader_index: int = 0,
    ) -> None:
        """Initialise the coordinator.

        Parameters
        ----------
        local_mpcs: List[MpcMld]
            List of local MLD based MPCs - one for each agent.
        """
        super().__init__(local_mpcs[0])
        self.n = len(local_mpcs)
        self.agents: list[MldAgent] = []
        for i in range(self.n):
            self.agents.append(MldAgent(local_mpcs[i]))

        self.nx_l = Vehicle.nx_l
        self.nu_l = Vehicle.nu_l
        self.vehicles = vehicles
        self.num_iters = event_iters
        self.leader_x = leader_x
        self.ts = ts
        self.N = N
        self.discrete_gears = discrete_gears
        self.leader_index = leader_index

        # store control and state guesses
        self.state_guesses = [np.zeros((self.nx_l, N + 1)) for i in range(self.n)]
        self.control_guesses = [np.zeros((self.nu_l, N)) for i in range(self.n)]
        self.gear_guesses = [np.zeros((self.nu_l, N)) for i in range(self.n)]

        self.solve_times = np.zeros((ep_len, 1))
        self.node_counts = np.zeros((ep_len, 1))
        # temp value used to sum up the itermediate solves over iterations
        self.temp_solve_time = 0
        self.temp_node_count = 0

    def get_control(self, state):
        [None] * self.n

        temp_costs = [None] * self.n
        for iter in range(self.num_iters):
            print(f"iter {iter + 1}")
            best_cost_dec = -float("inf")
            best_idx = -1  # gets set to an agent index if there is a cost improvement
            feasible_sol_flag = False
            for i in range(self.n):
                # get local initial condition and local initial guesses
                if i == 0:
                    x_l = state[self.nx_l * i : self.nx_l * (i + 2), :]
                    x_guess = np.vstack(
                        (self.state_guesses[i], self.state_guesses[i + 1])
                    )
                    u_guess = np.vstack(
                        (self.control_guesses[i], self.control_guesses[i + 1])
                    )
                elif i == self.n - 1:
                    x_l = state[self.nx_l * (i - 1) : self.nx_l * (i + 1), :]
                    x_guess = np.vstack(
                        (self.state_guesses[i - 1], self.state_guesses[i])
                    )
                    u_guess = np.vstack(
                        (self.control_guesses[i - 1], self.control_guesses[i])
                    )
                else:
                    x_l = state[self.nx_l * (i - 1) : self.nx_l * (i + 2), :]
                    x_guess = np.vstack(
                        (
                            self.state_guesses[i - 1],
                            self.state_guesses[i],
                            self.state_guesses[i + 1],
                        )
                    )
                    u_guess = np.vstack(
                        (
                            self.control_guesses[i - 1],
                            self.control_guesses[i],
                            self.control_guesses[i + 1],
                        )
                    )

                # set the constant predictions for neighbors of neighbors
                if i > 1:
                    self.agents[i].mpc.set_x_f2(self.state_guesses[i - 2])
                if i < self.n - 2:
                    self.agents[i].mpc.set_x_b2(self.state_guesses[i + 2])

                temp_costs[i] = self.agents[i].mpc.eval_cost(
                    x_guess, u_guess, discrete_gears=self.discrete_gears
                )
                _, _ = self.agents[i].get_control(x_l)
                new_cost = self.agents[i].get_predicted_cost()

                if new_cost < float("inf"):
                    feasible_sol_flag = True

                if (temp_costs[i] - new_cost > best_cost_dec) and (
                    temp_costs[i] - new_cost > threshold
                ):
                    best_cost_dec = temp_costs[i] - new_cost
                    best_idx = i

            if not feasible_sol_flag:
                raise RuntimeWarning(
                    "No feasible solution found for any event based agent."
                )
            # get solve times and node count
            self.temp_solve_time += max(
                [self.agents[i].run_time for i in range(self.n)]
            )
            self.temp_node_count = max(
                max([self.agents[i].node_count for i in range(self.n)]),
                self.temp_node_count,
            )

            # update state and control guesses based on the winner
            if best_idx >= 0:
                best_x = self.agents[best_idx].x_pred
                best_u = self.agents[best_idx].u_pred
                if self.discrete_gears:  # mpc has gears pred if it is gear mpc
                    best_gears = self.agents[best_idx].mpc.gears_pred
                if best_idx == 0:
                    self.state_guesses[0] = best_x[0:2, :]
                    self.state_guesses[1] = best_x[2:4, :]
                    self.control_guesses[0] = best_u[[0], :]
                    self.control_guesses[1] = best_u[[1], :]
                    if self.discrete_gears:
                        self.gear_guesses[0] = best_gears[[0], :]
                        self.gear_guesses[1] = best_gears[[1], :]
                elif best_idx == self.n - 1:
                    self.state_guesses[self.n - 2] = best_x[0:2, :]
                    self.state_guesses[self.n - 1] = best_x[2:4, :]
                    self.control_guesses[self.n - 2] = best_u[[0], :]
                    self.control_guesses[self.n - 1] = best_u[[1], :]
                    if self.discrete_gears:
                        self.gear_guesses[self.n - 2] = best_gears[[0], :]
                        self.gear_guesses[self.n - 1] = best_gears[[1], :]
                else:
                    self.state_guesses[best_idx - 1] = best_x[0:2, :]
                    self.state_guesses[best_idx] = best_x[2:4, :]
                    self.state_guesses[best_idx + 1] = best_x[4:6, :]
                    self.control_guesses[best_idx - 1] = best_u[[0], :]
                    self.control_guesses[best_idx] = best_u[[1], :]
                    self.control_guesses[best_idx + 1] = best_u[[2], :]
                    if self.discrete_gears:
                        self.gear_guesses[best_idx - 1] = best_gears[[0], :]
                        self.gear_guesses[best_idx] = best_gears[[1], :]
                        self.gear_guesses[best_idx + 1] = best_gears[[2], :]

            else:  # don't repeat the repetitions if no-one improved cost
                break

        if self.discrete_gears:
            return (
                np.vstack(
                    (
                        np.vstack(
                            [self.control_guesses[i][:, [0]] for i in range(self.n)]
                        ),
                        np.vstack(
                            [self.gear_guesses[i][:, [0]] for i in range(self.n)]
                        ),
                    )
                ),
                {},
            )
        else:
            return (
                np.vstack([self.control_guesses[i][:, [0]] for i in range(self.n)]),
                {},
            )

    def on_timestep_end(self, env: Env, episode: int, timestep: int) -> None:
        leader_x = self.leader_x[:, timestep : timestep + self.N + 1]
        self.agents[self.leader_index].mpc.set_leader_x(leader_x)
        if self.leader_index > 0:
            self.agents[self.leader_index - 1].mpc.set_leader_x(leader_x)
        if self.leader_index < self.n - 1:
            self.agents[self.leader_index + 1].mpc.set_leader_x(leader_x)

        # shift previous solutions to be initial guesses at next step
        for i in range(self.n):
            self.state_guesses[i] = np.concatenate(
                (self.state_guesses[i][:, 1:], self.state_guesses[i][:, -1:]),
                axis=1,
            )
            self.control_guesses[i] = np.concatenate(
                (self.control_guesses[i][:, 1:], self.control_guesses[i][:, -1:]),
                axis=1,
            )
            if self.discrete_gears:
                self.gear_guesses[i] = np.concatenate(
                    (self.gear_guesses[i][:, 1:], self.gear_guesses[i][:, -1:]),
                    axis=1,
                )

        self.solve_times[env.step_counter - 1, :] = self.temp_solve_time
        self.node_counts[env.step_counter - 1, :] = self.temp_node_count
        self.temp_solve_time = 0
        self.temp_node_count = 0

        return super().on_timestep_end(env, episode, timestep)

    def on_episode_start(self, env: Env, episode: int, state) -> None:
        leader_x = self.leader_x[:, 0 : self.N + 1]
        self.agents[self.leader_index].mpc.set_leader_x(leader_x)
        if self.leader_index > 0:
            self.agents[self.leader_index - 1].mpc.set_leader_x(leader_x)
        if self.leader_index < self.n - 1:
            self.agents[self.leader_index + 1].mpc.set_leader_x(leader_x)

        # initialise first step guesses with extrapolating positions
        for i in range(self.n):
            xl = env.x[
                self.nx_l * i : self.nx_l * (i + 1), :
            ]  # initial local state for vehicle i
            self.state_guesses[i] = self.extrapolate_position(xl[0, :], xl[1, :])
            if not self.discrete_gears:
                self.control_guesses[i] = self.vehicles[i].get_u_for_constant_vel(
                    xl[1, 0]
                ) * np.ones((self.nu_l, self.N))
            else:
                j = self.vehicles[i].get_gear_from_velocity(xl[1, 0])
                self.gear_guesses[i] = j * np.ones((self.nu_l, self.N))
                self.control_guesses[i] = self.vehicles[i].get_u_for_constant_vel(
                    xl[1, 0], j
                ) * np.ones((self.nu_l, self.N))

        return super().on_episode_start(env, episode, state)

    def extrapolate_position(self, initial_pos, initial_vel):
        x_pred = np.zeros((self.nx_l, self.N + 1))
        x_pred[0, [0]] = initial_pos
        x_pred[1, [0]] = initial_vel
        for k in range(self.N):
            x_pred[0, [k + 1]] = x_pred[0, [k]] + self.ts * x_pred[1, [k]]
            x_pred[1, [k + 1]] = x_pred[1, [k]]
        return x_pred


def simulate(
    sim: Sim,
    event_iters: int,
    save: bool = False,
    plot: bool = True,
    seed: int = 2,
    thread_limit: int | None = None,
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
                leader_index=leader_index,
                ep_len=sim.ep_len,
            ),
            max_episode_steps=ep_len,
        )
    )

    # mpcs
    if sim.vehicle_model_type == "pwa_gear":
        mpc_class = LocalMpc
        discrete_gears = False
    elif sim.vehicle_model_type == "pwa_friction":
        mpc_class = LocalMpcGear
        discrete_gears = True
    elif sim.vehicle_model_type == "nonlinear":
        mpc_class = LocalMpcNonlinearGear
        discrete_gears = True
    else:
        raise ValueError(f"{sim.vehicle_model_type} is not a valid vehicle model type.")

    mpcs = [
        mpc_class(
            N,
            systems=(
                systems[:2]
                if i == 0
                else (systems[-2:] if i == n - 1 else systems[i - 1 : i + 2])
            ),
            num_vehicles_in_front=i if i < 2 else 2,
            num_vehicles_behind=(n - 1) - i if i > n - 3 else 2,
            rel_leader_index=(
                -1
                if i == leader_index - 1
                else (
                    0 if i == leader_index else (1 if i == leader_index + 1 else None)
                )
            ),
            spacing_policy=spacing_policy,
            thread_limit=thread_limit,
        )
        for i in range(n)
    ]
    # agent
    agent = TrackingEventBasedCoordinator(
        mpcs,
        vehicles=platoon.get_vehicles(),
        ep_len=ep_len,
        N=N,
        leader_x=leader_x,
        discrete_gears=discrete_gears,
        ts=ts,
        event_iters=event_iters,
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
            f"event_{event_iters}_{sim.id}_seed_{seed}" + ".pkl",
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
    simulate(Sim(), plot=True, event_iters=4, seed=0, leader_index=0)
