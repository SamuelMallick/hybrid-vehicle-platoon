import pickle
import sys

import gurobipy as gp
import numpy as np
from ACC_env import CarFleet
from ACC_model import ACC
from gymnasium import Env
from gymnasium.wrappers import TimeLimit
from mpcrl.wrappers.envs import MonitorEpisodes
from mpcs.mpc_gear import MpcGear
from plot_fleet import plot_fleet
from scipy.linalg import block_diag

from dmpcpwa.agents.mld_agent import MldAgent
from dmpcpwa.mpc.mpc_mld_cent_decup import MpcMldCentDecup

np.random.seed(2)

DEBUG = False

PLOT = True
SAVE = False

n = 3  # num cars
N = 5  # controller horizon
COST_2_NORM = True
DISCRETE_GEARS = False
HOMOGENOUS = True
LEADER_TRAJ = 2  # "1" - constant velocity leader traj. Vehicles start from random ICs. "2" - accelerating leader traj. Vehicles start in perfect platoon.

num_iters = 4
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
    num_iters = int(sys.argv[7])

random_ICs = False
if LEADER_TRAJ == 1:
    random_ICs = True

threshold = 10  # cost improvement must be more than this to consider communication
follow_bias = 1  # a slight bias added to the cost to favour following the vehicle in front in case of tiebrake in cost improvements

ep_len = 100  # length of episode (sim len)

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


class LocalMpc(MpcMldCentDecup):
    """Mpc for a vehicle with a car in front and behind. Local state has car
    is organised with x = [x_front, x_me, x_back]."""

    def __init__(
        self, systems: list[dict], n: int, N: int, pos_in_fleet: int, num_vehicles: int
    ) -> None:
        super().__init__(systems, n, N)
        self.setup_cost_and_constraints(self.u, pos_in_fleet, num_vehicles)

    def setup_cost_and_constraints(self, u, pos_in_fleet, num_vehicles):
        if COST_2_NORM:
            cost_func = self.min_2_norm
        else:
            cost_func = self.min_1_norm

        self.pos_in_fleet = pos_in_fleet
        self.num_vehicles = num_vehicles
        # the index of the local vehicles position is different if it is the leader or trailer
        if pos_in_fleet == 1:
            my_index = 0
            b_index = 2
            self.b_index = b_index
        elif pos_in_fleet == num_vehicles:
            my_index = 2
            f_index = 0
            self.f_index = f_index
        else:
            my_index = 2
            f_index = 0
            b_index = 4
            self.f_index = f_index
            self.b_index = b_index
        self.my_index = my_index

        # constraints and slacks for cars in front
        if pos_in_fleet > 1:
            self.s_front = self.mpc_model.addMVar(
                (1, N + 1), lb=0, ub=float("inf"), name="s_front"
            )
            for k in range(N + 1):
                self.mpc_model.addConstr(
                    self.x[f_index, [k]] - self.x[my_index, [k]]
                    >= d_safe - self.s_front[:, [k]],
                    name=f"safety_ahead_{k}",
                )
            if pos_in_fleet > 2:
                self.s_front_2 = self.mpc_model.addMVar(
                    (1, N + 1), lb=0, ub=float("inf"), name="s_front_2"
                )
                self.x_front_2 = self.mpc_model.addMVar(
                    (nx_l, N + 1), lb=0, ub=0, name="x_front_2"
                )
                for k in range(N + 1):
                    self.mpc_model.addConstr(
                        self.x_front_2[0, [k]] - self.x[f_index, [k]]
                        >= d_safe - self.s_front_2[:, [k]],
                        name=f"safety_ahead_2_{k}",
                    )
        if pos_in_fleet <= 2:  # leader and its follower
            self.ref_traj = self.mpc_model.addMVar(
                (nx_l, N + 1), lb=0, ub=0, name="ref_traj"
            )

        # constraints and slacks for cars in back
        if num_vehicles - pos_in_fleet >= 1:
            self.s_back = self.mpc_model.addMVar(
                (1, N + 1), lb=0, ub=float("inf"), name="s_back"
            )
            for k in range(N + 1):
                self.mpc_model.addConstr(
                    self.x[my_index, [k]] - self.x[b_index, [k]]
                    >= d_safe - self.s_back[:, [k]],
                    name=f"safety_back_{k}",
                )

            if num_vehicles - pos_in_fleet >= 2:
                self.s_back_2 = self.mpc_model.addMVar(
                    (1, N + 1), lb=0, ub=float("inf"), name="s_back_2"
                )
                # fixed state of car 2 back
                self.x_back_2 = self.mpc_model.addMVar(
                    (nx_l, N + 1), lb=0, ub=0, name="x_back_2"
                )

                for k in range(N + 1):
                    self.mpc_model.addConstr(
                        self.x[b_index, [k]] - self.x_back_2[0, [k]]
                        >= d_safe - self.s_back_2[:, [k]],
                        name=f"safety_back_2_{k}",
                    )

        # accel cnstrs
        for k in range(N):
            for i in range(u.shape[0]):
                self.mpc_model.addConstr(
                    self.x[2 * i + 1, [k + 1]] - self.x[2 * i + 1, [k]]
                    <= acc.a_acc * acc.ts,
                    name=f"acc_{i}_{k}",
                )
                self.mpc_model.addConstr(
                    self.x[2 * i + 1, [k + 1]] - self.x[2 * i + 1, [k]]
                    >= acc.a_dec * acc.ts,
                    name=f"dec_{i}_{k}",
                )

        # set local cost
        obj = 0
        # front position tracking portions of cost
        if pos_in_fleet > 1:
            for k in range(N + 1):
                obj += follow_bias * cost_func(
                    (
                        self.x[my_index : my_index + 2, [k]]
                        - self.x[f_index : f_index + 2, [k]]
                        - sep
                    ),
                    Q_x_l,
                )
                +w * self.s_front[:, [k]]
            if pos_in_fleet > 2:
                for k in range(N + 1):
                    obj += follow_bias * (
                        cost_func(
                            (
                                self.x[f_index : f_index + 2, [k]]
                                - self.x_front_2[:, [k]]
                                - sep
                            ),
                            Q_x_l,
                        )
                        + w * self.s_front_2[:, k]
                    )
        if pos_in_fleet == 1:  # leader
            for k in range(N + 1):
                obj += follow_bias * cost_func(
                    (
                        self.x[my_index : my_index + 2, [k]]
                        - self.ref_traj[:, [k]]
                        - np.zeros((nx_l, 1))
                    ),
                    Q_x_l,
                )
        if pos_in_fleet == 2:  # follower of leader
            for k in range(N + 1):
                obj += follow_bias * cost_func(
                    (
                        self.x[f_index : f_index + 2, [k]]
                        - self.ref_traj[:, [k]]
                        - np.zeros((nx_l, 1))
                    ),
                    Q_x_l,
                )

        # back position tracking
        if num_vehicles - pos_in_fleet >= 1:
            for k in range(N + 1):
                obj += (
                    cost_func(
                        (
                            self.x[b_index : b_index + 2, [k]]
                            - self.x[my_index : my_index + 2, [k]]
                            - sep
                        ),
                        Q_x_l,
                    )
                    + w * self.s_back[:, k]
                )
            if num_vehicles - pos_in_fleet >= 2:
                for k in range(N + 1):
                    obj += (
                        cost_func(
                            (
                                self.x_back_2[:, [k]]
                                - self.x[b_index : b_index + 2, [k]]
                                - sep
                            ),
                            Q_x_l,
                        )
                        + w * self.s_back_2[:, k]
                    )

        # control penalty in cost
        for i in range(u.shape[0]):
            for k in range(N):
                obj += cost_func(u[i, [k]].reshape(1, 1), Q_u_l)
                if k < N - 1:
                    obj += cost_func(
                        u[i, [k + 1]].reshape(1, 1) - u[i, [k]].reshape(1, 1), Q_du_l
                    )

        self.mpc_model.setObjective(obj, gp.GRB.MINIMIZE)

    def set_leader_traj(self, ref_traj):
        for k in range(N + 1):
            self.ref_traj[:, [k]].lb = ref_traj[:, [k]]
            self.ref_traj[:, [k]].ub = ref_traj[:, [k]]

    def set_x_front_2(self, x_front_2):
        for k in range(N + 1):
            self.x_front_2[:, [k]].lb = x_front_2[:, [k]]
            self.x_front_2[:, [k]].ub = x_front_2[:, [k]]

    def set_x_back_2(self, x_back_2):
        for k in range(N + 1):
            self.x_back_2[:, [k]].lb = x_back_2[:, [k]]
            self.x_back_2[:, [k]].ub = x_back_2[:, [k]]

    def eval_cost(self, x, u):
        # set the bounds of the vars in the model to fix the vals
        for k in range(
            N
        ):  # we dont constain the N+1th state, as it is defined by shifted control
            self.x[:, [k]].ub = x[:, [k]]
            self.x[:, [k]].lb = x[:, [k]]
        if DISCRETE_GEARS:
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

    def solve_mpc(self, state):
        # the solve method is overridden so that the bounds on the vars are set back to normal before solving.
        self.x.ub = float("inf")
        self.x.lb = -float("inf")
        if DISCRETE_GEARS:
            self.u_g.ub = float("inf")
            self.u_g.lb = -float("inf")
        else:
            self.u.ub = float("inf")
            self.u.lb = -float("inf")
        return super().solve_mpc(state)


class LocalMpcGear(LocalMpc, MpcMldCentDecup, MpcGear):
    def __init__(
        self, systems: list[dict], n: int, N: int, pos_in_fleet: int, num_vehicles: int
    ) -> None:
        MpcMldCentDecup.__init__(self, systems, n, N)
        F = block_diag(*([systems[0]["F"]] * n))
        G = np.vstack([systems[0]["G"]] * n)
        self.setup_gears(N, acc, F, G)
        self.setup_cost_and_constraints(self.u_g, pos_in_fleet, num_vehicles)


class TrackingEventBasedCoordinator(MldAgent):
    def __init__(
        self,
        local_mpcs: list[LocalMpc],
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

        # store control and state guesses
        self.state_guesses = [np.zeros((nx_l, N + 1)) for i in range(n)]
        self.control_guesses = [np.zeros((nu_l, N)) for i in range(n)]
        self.gear_guesses = [np.zeros((nu_l, N)) for i in range(n)]

        self.solve_times = np.zeros((ep_len, 1))
        self.node_counts = np.zeros((ep_len, 1))
        # temp value used to sum up the itermediate solves over iterations
        self.temp_solve_time = 0
        self.temp_node_count = 0

    def get_control(self, state):
        [None] * self.n

        temp_costs = [None] * self.n
        for iter in range(num_iters):
            print(f"iter {iter + 1}")
            best_cost_dec = -float("inf")
            best_idx = -1  # gets set to an agent index if there is a cost improvement
            for i in range(self.n):
                # get local initial condition and local initial guesses
                if i == 0:
                    x_l = state[nx_l * i : nx_l * (i + 2), :]
                    x_guess = np.vstack(
                        (self.state_guesses[i], self.state_guesses[i + 1])
                    )
                    u_guess = np.vstack(
                        (self.control_guesses[i], self.control_guesses[i + 1])
                    )
                elif i == n - 1:
                    x_l = state[nx_l * (i - 1) : nx_l * (i + 1), :]
                    x_guess = np.vstack(
                        (self.state_guesses[i - 1], self.state_guesses[i])
                    )
                    u_guess = np.vstack(
                        (self.control_guesses[i - 1], self.control_guesses[i])
                    )
                else:
                    x_l = state[nx_l * (i - 1) : nx_l * (i + 2), :]
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
                    self.agents[i].mpc.set_x_front_2(self.state_guesses[i - 2])
                if i < n - 2:
                    self.agents[i].mpc.set_x_back_2(self.state_guesses[i + 2])

                temp_costs[i] = self.agents[i].mpc.eval_cost(x_guess, u_guess)
                self.agents[i].get_control(x_l)
                new_cost = self.agents[i].get_predicted_cost()
                if (temp_costs[i] - new_cost > best_cost_dec) and (
                    temp_costs[i] - new_cost > threshold
                ):
                    best_cost_dec = temp_costs[i] - new_cost
                    best_idx = i

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
                if DISCRETE_GEARS:  # mpc has gears pred if it is gear mpc
                    best_gears = self.agents[best_idx].mpc.gears_pred
                if best_idx == 0:
                    self.state_guesses[0] = best_x[0:2, :]
                    self.state_guesses[1] = best_x[2:4, :]
                    self.control_guesses[0] = best_u[[0], :]
                    self.control_guesses[1] = best_u[[1], :]
                    if DISCRETE_GEARS:
                        self.gear_guesses[0] = best_gears[[0], :]
                        self.gear_guesses[1] = best_gears[[1], :]
                elif best_idx == n - 1:
                    self.state_guesses[n - 2] = best_x[0:2, :]
                    self.state_guesses[n - 1] = best_x[2:4, :]
                    self.control_guesses[n - 2] = best_u[[0], :]
                    self.control_guesses[n - 1] = best_u[[1], :]
                    if DISCRETE_GEARS:
                        self.gear_guesses[n - 2] = best_gears[[0], :]
                        self.gear_guesses[n - 1] = best_gears[[1], :]
                else:
                    self.state_guesses[best_idx - 1] = best_x[0:2, :]
                    self.state_guesses[best_idx] = best_x[2:4, :]
                    self.state_guesses[best_idx + 1] = best_x[4:6, :]
                    self.control_guesses[best_idx - 1] = best_u[[0], :]
                    self.control_guesses[best_idx] = best_u[[1], :]
                    self.control_guesses[best_idx + 1] = best_u[[2], :]
                    if DISCRETE_GEARS:
                        self.gear_guesses[best_idx - 1] = best_gears[[0], :]
                        self.gear_guesses[best_idx] = best_gears[[1], :]
                        self.gear_guesses[best_idx + 1] = best_gears[[2], :]

            else:  # don't repeat the repetitions if no-one improved cost
                break

        # debugging
        if DEBUG:
            cost_inc = 0
            for i in range(n):
                local_x = self.state_guesses[i]
                local_u = self.control_guesses[i]
                for k in range(N):
                    if i == 0:
                        front = self.agents[0].mpc.ref_traj.X
                        sep_temp = np.zeros((2, 1))
                    else:
                        front = self.state_guesses[i - 1]
                        sep_temp = sep
                    cost_inc += (local_x[:, k] - front[:, k] - sep_temp.T) @ Q_x_l @ (
                        local_x[:, [k]] - front[:, [k]] - sep_temp
                    ) + local_u[:, k] @ Q_u_l @ local_u[:, [k]]
                cost_inc += (
                    (local_x[:, N] - front[:, N] - sep_temp.T)
                    @ Q_x_l
                    @ (local_x[:, [N]] - front[:, [N]] - sep_temp)
                )

        if DISCRETE_GEARS:
            return np.vstack(
                (
                    np.vstack([self.control_guesses[i][:, [0]] for i in range(n)]),
                    np.vstack([self.gear_guesses[i][:, [0]] for i in range(n)]),
                )
            )
        else:
            return np.vstack([self.control_guesses[i][:, [0]] for i in range(n)])

    def on_timestep_end(self, env: Env, episode: int, timestep: int) -> None:
        self.agents[0].mpc.set_leader_traj(leader_state[:, timestep : timestep + N + 1])
        self.agents[1].mpc.set_leader_traj(leader_state[:, timestep : timestep + N + 1])

        # shift previous solutions to be initial guesses at next step
        for i in range(n):
            self.state_guesses[i] = np.concatenate(
                (self.state_guesses[i][:, 1:], self.state_guesses[i][:, -1:]),
                axis=1,
            )
            self.control_guesses[i] = np.concatenate(
                (self.control_guesses[i][:, 1:], self.control_guesses[i][:, -1:]),
                axis=1,
            )
            if DISCRETE_GEARS:
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
        self.agents[0].mpc.set_leader_traj(leader_state[:, 0 : N + 1])
        self.agents[1].mpc.set_leader_traj(leader_state[:, 0 : N + 1])

        # initialise first step guesses with extrapolating positions
        for i in range(self.n):
            xl = env.x[
                nx_l * i : nx_l * (i + 1), :
            ]  # initial local state for vehicle i
            self.state_guesses[i] = self.extrapolate_position(xl[0, :], xl[1, :])
            self.control_guesses[i] = acc.get_u_for_constant_vel(xl[1, :]) * np.ones(
                (nu_l, N)
            )
            if DISCRETE_GEARS:
                # for a gear guess we use twa mapping from speed to gear
                self.gear_guesses[i] = acc.get_pwa_gear_from_speed(xl[1, :]) * np.ones(
                    (nu_l, N)
                )

        return super().on_episode_start(env, episode, state)

    def extrapolate_position(self, initial_pos, initial_vel):
        x_pred = np.zeros((nx_l, N + 1))
        x_pred[0, [0]] = initial_pos
        x_pred[1, [0]] = initial_vel
        for k in range(N):
            x_pred[0, [k + 1]] = x_pred[0, [k]] + acc.ts * x_pred[1, [k]]
            x_pred[1, [k + 1]] = x_pred[1, [k]]
        return x_pred


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
    mpc_class = LocalMpc
    if HOMOGENOUS:  # by not passing the index all systems are the same
        systems = [acc.get_pwa_system() for i in range(n)]
    else:
        systems = [acc.get_pwa_system(i) for i in range(n)]

# coordinator
local_mpcs: list[LocalMpc] = []
for i in range(n):
    # create mpcs
    if i == 0:
        local_mpcs.append(mpc_class([systems[0], systems[1]], 2, N, i + 1, n))
    elif i == n - 1:
        local_mpcs.append(mpc_class([systems[n - 2], systems[n - 1]], 2, N, i + 1, n))
    else:
        local_mpcs.append(
            mpc_class([systems[i - 1], systems[i], systems[i + 1]], 3, N, i + 1, n)
        )

agent = TrackingEventBasedCoordinator(local_mpcs)

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
print(f"Mem: {max(agent.node_counts)}")

if PLOT:
    plot_fleet(n, X, U, R, leader_state, violations=env.unwrapped.viol_counter[0])

if SAVE:
    with open(
        f"event{num_iters}_n_{n}_N_{N}_Q_{COST_2_NORM}_DG_{DISCRETE_GEARS}_HOM_{HOMOGENOUS}_LT_{LEADER_TRAJ}"
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
