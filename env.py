from typing import Any

import gymnasium as gym
import numpy as np
import numpy.typing as npt

from misc.leader_trajectory import ConstantVelocityLeaderTrajectory, LeaderTrajectory
from misc.spacing_policy import ConstantSpacingPolicy, SpacingPolicy
from models import Platoon


class PlatoonEnv(gym.Env[npt.NDArray[np.floating], npt.NDArray[np.floating]]):
    """An env for a platoon of non-linear hybrid vehicles who track each other."""

    # tracking costs
    Q_x = np.diag([1, 0.1])  # penalty of state tracking error
    Q_u = 1 * np.eye(1)  # penalty on control effort
    Q_du = 0 * np.eye(1)  # penalty on variation in control effort

    # local state and control dimension of vehicles in the platoon
    nx_l = Platoon.nx_l
    nu_l = Platoon.nu_l

    step_counter = 0
    viol_counter = []  # count constraint violations

    def __init__(
        self,
        n: int,
        platoon: Platoon,
        ep_len: int = 100,
        ts: float = 1,
        leader_trajectory: LeaderTrajectory = ConstantVelocityLeaderTrajectory(
            p=3000, v=20, trajectory_len=150, ts=1
        ),
        spacing_policy: SpacingPolicy = ConstantSpacingPolicy(50),
        d_safe: float = 25,
        start_from_platoon: bool = False,
        quadratic_cost: bool = True,
    ) -> None:
        super().__init__()

        self.platoon = platoon
        self.ts = ts
        self.n = n
        self.ep_len = ep_len
        self.d_safe = d_safe
        self.start_from_platoon = start_from_platoon
        self.leader_trajectory = leader_trajectory
        self.spacing_policy = spacing_policy
        if quadratic_cost:
            self.cost_func = self.quad_cost
        else:
            self.cost_func = self.lin_cost

        self.previous_action: np.ndarray | None = (
            None  # store previous action to penalise variation
        )

    def reset(
        self,
        *,
        seed: int = None,
        options: dict[str, Any] = None,
    ) -> tuple[npt.NDArray[np.floating], dict[str, Any]]:
        """Resets the state of the platoon."""
        super().reset(seed=seed, options=options)
        self.leader_x = self.leader_trajectory.get_leader_trajectory()
        self.x = np.tile(np.array([[0], [0]]), (self.n, 1))

        np.random.seed(2)
        # create once 100 random starting states
        starting_velocities = [30] + [
            20 * np.random.random() + 5 for i in range(100)
        ]  # starting velocities between 5-40 ms-1
        # starting positions between 0-1000 meters, with some forced spacing
        front_pos = 3000.0
        spread = 50
        spacing = 50
        starting_positions = [front_pos]
        for i in range(1, 100):
            starting_positions.append(
                -spread * np.random.random() + starting_positions[-1] - spacing
            )
        if not self.start_from_platoon:
            # order the agents by starting distance
            for i in range(self.n):
                init_pos = max(starting_positions)
                self.x[i * self.nx_l, :] = init_pos
                self.x[i * self.nx_l + 1, :] = starting_velocities[i]
                starting_positions.remove(init_pos)

        else:  # if not random, the vehicles start in perfect platoon with leader on trajectory
            for i in range(self.n):
                self.x[i * self.nx_l : self.nx_l * (i + 1), :] = self.leader_x[
                    :, [0]
                ] + i * self.spacing_policy.spacing(self.leader_x[:, [0]])

        self.step_counter = 0
        self.viol_counter.append(np.zeros(self.ep_len))
        return self.x, {}

    def quad_cost(self, x: np.ndarray, Q: np.ndarray):
        """Returns x'Qx"""
        return x.T @ Q @ x

    def lin_cost(self, x: np.ndarray, Q: np.ndarray):
        """Return sum_i |Q_i x_i|."""
        return np.linalg.norm(Q @ x, ord=1)

    def get_stage_cost(
        self, state: npt.NDArray[np.floating], action: npt.NDArray[np.floating]
    ) -> float:
        """Computes the tracking stage cost."""
        if (
            self.previous_action is None
        ):  # for the first time step the variation penalty will be zero
            self.previous_action = action

        # split global vars into list of local
        x = np.split(state, self.n, axis=0)
        u = np.split(action, self.n, axis=0)
        u_p = np.split(self.previous_action, self.n, axis=0)

        cost = 0
        # tracking cost
        cost += self.cost_func(
            x[0] - self.leader_x[:, [self.step_counter]], self.Q_x
        )  # first vehicle tracking leader trajectory
        cost += sum(
            [
                self.cost_func(
                    x[i] - x[i - 1] - (self.spacing_policy.spacing(x[i])),
                    self.Q_x,
                )
                for i in range(1, self.n)
            ]
        )
        # control effort cost
        cost += sum([self.cost_func(u[i], self.Q_u) for i in range(self.n)])
        # control variation cost
        cost += sum([self.cost_func(u[i] - u_p[i], self.Q_du) for i in range(self.n)])

        # check for constraint violations
        if any([x[i][0, 0] - x[i + 1][0, 0] < self.d_safe for i in range(self.n - 1)]):
            self.viol_counter[-1][self.step_counter] = 100

        self.previous_action = action
        return cost

    def step(
        self, action: np.ndarray
    ) -> tuple[npt.NDArray[np.floating], float, bool, bool, dict[str, Any]]:
        """Steps the platoon system."""
        if action.shape != (self.n * self.nu_l, 1) and action.shape != (
            2 * self.n * self.nu_l,
            1,
        ):
            raise ValueError(
                f"Expected action of size {(self.n*self.nu_l, 1)} (no gears) or {(2*self.n*self.nu_l, 1)} (with gears). Got {action.shape}"
            )
        if (
            action.shape[0] == 2 * self.n * self.nu_l
        ):  # this action contains n gear choices aswell as continuous throttle vals
            u = action[: self.n * self.nu_l, :]
            j = action[self.n * self.nu_l :, :]
        else:
            u = action
            j = np.zeros((self.n, 1))
            for i in range(self.n):
                j[i, :] = self.platoon.get_gear_from_vehicle_velocity(
                    i, self.x[2 * i + 1, 0]
                )

        r = self.get_stage_cost(self.x, u)
        x_new = self.platoon.step_platoon(self.x, u, j, self.ts)
        self.x = x_new

        self.step_counter += 1
        print(f"step {self.step_counter}")
        return x_new, r, False, False, {}
