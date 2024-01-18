from typing import Any

import casadi as cs
import gymnasium as gym
import numpy as np
import numpy.typing as npt
from ACC_model import ACC


class CarFleet(gym.Env[npt.NDArray[np.floating], npt.NDArray[np.floating]]):
    """A fleet of non-linear hybrid vehicles who track each other."""

    step_counter = 0
    viol_counter = []  # count constraint violations for each ep

    def __init__(
        self,
        acc: ACC,
        n: int,
        ep_len: int,
        L2_norm_cost=True,
        homogenous=True,
        random_ICs=False,
    ) -> None:
        self.acc = acc
        self.nx_l = acc.nx_l
        self.nu_l = acc.nu_l
        self.nu_l = acc.nu_l
        self.Q_x_l = acc.Q_x_l
        self.Q_u_l = acc.Q_u_l
        self.Q_du_l = acc.Q_du_l
        self.sep = acc.sep
        self.leader_state = acc.get_leader_state()
        self.n = n
        self.ep_len = ep_len

        self.homogenous = homogenous
        self.random_ICs = random_ICs
        self.L2_norm_cost = L2_norm_cost

        self.previous_action = None  # store previous action to penalise variation

        super().__init__()

    def reset(
        self,
        *,
        seed: int = None,
        options: dict[str, Any] = None,
    ) -> tuple[npt.NDArray[np.floating], dict[str, Any]]:
        """Resets the state of the LTI system."""
        super().reset(seed=seed, options=options)

        self.x = np.tile(np.array([[0], [0]]), (self.n, 1))

        if self.random_ICs:
            starting_velocities = [30] + [
                20 * np.random.random() + 5 for i in range(self.n - 1)
            ]  # starting velocities between 5-40 ms-1

            # starting positions between 0-1000 meters, with some forced spacing
            front_pos = 3000.0
            spread = 50
            spacing = 50
            starting_positions = [front_pos]
            # starting_positions = [front_pos - spread * np.random.random()]
            for i in range(1, self.n):
                starting_positions.append(
                    -spread * np.random.random() + starting_positions[-1] - spacing
                )

            for i in range(self.n):
                init_pos = max(
                    starting_positions
                )  # order the agents by starting distance
                self.x[i * self.nx_l, :] = init_pos
                self.x[i * self.nx_l + 1, :] = starting_velocities[i]
                starting_positions.remove(init_pos)

        else:  # if not random, the vehicles start in perfect platoon with leader on trajectory
            for i in range(self.n):
                self.x[i * self.nx_l : self.nx_l * (i + 1), :] = (
                    self.leader_state[:, [0]] + i * self.sep
                )

        self.step_counter = 0
        self.viol_counter.append(np.zeros(self.ep_len))
        return self.x, {}

    def get_stage_cost(
        self, state: npt.NDArray[np.floating], action: npt.NDArray[np.floating]
    ) -> float:
        """Computes the stage cost `L(s,a)`."""

        if (
            self.previous_action is None
        ):  # for the first time step the variation penalty will be zero
            self.previous_action = action

        cost = 0
        for i in range(self.n):
            local_state = state[self.nx_l * i : self.nx_l * (i + 1), :]
            local_action = action[self.nu_l * i : self.nu_l * (i + 1), :]
            local_prev_action = self.previous_action[
                self.nu_l * i : self.nu_l * (i + 1), :
            ]
            if i == 0:
                # first car tracks leader
                follow_state = self.leader_state[:, [self.step_counter]]
                if self.L2_norm_cost:
                    cost += (
                        (local_state - follow_state).T
                        @ self.Q_x_l
                        @ (local_state - follow_state)
                        + local_action.T @ self.Q_u_l @ local_action
                        + (local_action - local_prev_action).T
                        @ self.Q_du_l
                        @ (local_action - local_prev_action)
                    )
                else:
                    cost += (
                        np.linalg.norm(self.Q_x_l @ (local_state - follow_state), ord=1)
                        + np.linalg.norm(self.Q_u_l @ local_action, ord=1)
                        + np.linalg.norm(
                            self.Q_du_l @ (local_action - local_prev_action), ord=1
                        )
                    )
            else:
                # other cars follow the next car
                follow_state = state[self.nx_l * (i - 1) : self.nx_l * (i), :]
                if self.L2_norm_cost:
                    cost += (
                        (local_state - follow_state - self.sep).T
                        @ self.Q_x_l
                        @ (local_state - follow_state - self.sep)
                        + local_action.T @ self.Q_u_l @ local_action
                        + (local_action - local_prev_action).T
                        @ self.Q_du_l
                        @ (local_action - local_prev_action)
                    )
                else:
                    cost += (
                        np.linalg.norm(
                            self.Q_x_l @ (local_state - follow_state - self.sep), ord=1
                        )
                        + np.linalg.norm(self.Q_u_l @ local_action, ord=1)
                        + np.linalg.norm(
                            self.Q_du_l @ (local_action - local_prev_action), ord=1
                        )
                    )

            # check for constraint violations
            if i < self.n - 1:
                local_state_behind = state[self.nx_l * (i + 1) : self.nx_l * (i + 2), :]
                if local_state[0] - local_state_behind[0] < self.acc.d_safe:
                    self.viol_counter[-1][self.step_counter] = 100

        self.previous_action = action

        return cost

    def step(
        self, action: cs.DM
    ) -> tuple[npt.NDArray[np.floating], float, bool, bool, dict[str, Any]]:
        """Steps the fleet system."""

        action = action.full()

        if (
            action.shape[0] > self.n
        ):  # this action contains n gear choices aswell as continuous throttle vals
            cont_action = action[: self.n, :]
            disc_action = action[self.n :, :]
        else:
            cont_action = action
            disc_action = np.zeros((self.n, 1))
            for i in range(self.n):
                disc_action[i, :] = self.acc.get_pwa_gear_from_speed(
                    self.x[2 * i + 1, :]
                )

        r = self.get_stage_cost(self.x, cont_action)
        x_new = self.acc.step_car_dynamics_nl(
            self.x, cont_action, disc_action, self.n, self.acc.ts, homog=self.homogenous
        )
        self.x = x_new

        self.step_counter += 1
        print(f"step {self.step_counter}")
        return x_new, r, False, False, {}
