from typing import Literal

import numpy as np

from misc.leader_trajectory import (
    ConstantVelocityLeaderTrajectory,
    StopAndGoLeaderTrajectory,
    VolatileTrajectory,
)
from misc.spacing_policy import ConstantSpacingPolicy, ConstantTimePolicy

np.random.seed(2)
random_masses = np.random.uniform(700, 900, 100).tolist()


class Params:
    Q_x = np.diag([1, 0.1])  # penalty of state tracking error
    Q_u = 1 * np.eye(1)  # penalty on control effort
    Q_du = 0 * np.eye(1)  # penalty on variation in control effort
    w = 1e4  # penalty on slack variables
    ts = 1
    a_acc = 2.5  # acceleration limit
    a_dec = -2  # deceleration limit
    d_safe = 25


class Sim:
    real_vehicle_as_reference = True
    vehicle_model_type: Literal["nonlinear", "pwa_friction", "pwa_gear"] = "pwa_gear"
    start_from_platoon: bool = True
    quadratic_cost: bool = True
    n = 4
    N = 8
    ep_len = 100
    # spacing_policy = ConstantSpacingPolicy(50)
    # leader_trajectory = ConstantVelocityLeaderTrajectory(
    #     p=3000, v=20, trajectory_len=ep_len + 50, ts=Params.ts
    # )
    leader_trajectory = VolatileTrajectory(
        p=3100, trajectory_len=ep_len + 50, ts=Params.ts
    )
    spacing_policy = ConstantTimePolicy(10, 3)
    # leader_trajectory = StopAndGoLeaderTrajectory(
    #     p=3100,
    #     vh=20,
    #     vl=10,
    #     vf=30,
    #     v_change_steps=[20, 25],
    #     trajectory_len=ep_len + 50,
    #     ts=Params.ts,
    # )
    masses = None
    id = f"default_n_{n}"


class Sim_n_task_1(Sim):
    def __init__(self, n: int) -> None:
        super().__init__()
        self.n = n
        self.id = f"task_1_n_{n}"
        self.spacing_policy = ConstantSpacingPolicy(50)
        self.leader_trajectory = ConstantVelocityLeaderTrajectory(
            p=3100, v=20, trajectory_len=self.ep_len + 50, ts=Params.ts
        )


class Sim_n_task_2(Sim):
    def __init__(self, n: int) -> None:
        super().__init__()
        self.n = n
        self.id = f"task_2_n_{n}"
        self.spacing_policy = ConstantTimePolicy(10, 3)
        self.leader_trajectory = StopAndGoLeaderTrajectory(
            p=3100,
            vh=20,
            vl=10,
            vf=30,
            v_change_steps=[40, 100],
            trajectory_len=self.ep_len + 50,
            ts=Params.ts,
        )
        self.masses = random_masses[:n]
