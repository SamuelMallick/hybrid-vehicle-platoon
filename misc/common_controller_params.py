from typing import Literal

import numpy as np

from misc.leader_trajectory import StopAndGoLeaderTrajectory
from misc.spacing_policy import ConstantTimePolicy


class Sim:
    vehicle_model_type: Literal["nonlinear", "pwa_friction", "pwa_gear"] = "pwa_gear"
    start_from_platoon: bool = True
    quadratic_cost: bool = True
    n = 4
    N = 5
    ep_len = 100
    # spacing_policy = ConstantSpacingPolicy(50)
    # leader_trajectory = ConstantVelocityLeaderTrajectory(
    #     p=3000, v=20, trajectory_len=ep_len + 50, ts=ts
    # )
    spacing_policy = ConstantTimePolicy(10, 3)
    leader_trajectory = StopAndGoLeaderTrajectory(
        p=3000, vh=30, vl=10, v_change_steps=[10, 50], trajectory_len=ep_len + 50, ts=ts
    )

class Params:
    Q_x = np.diag([1, 0.1])  # penalty of state tracking error
    Q_u = 1 * np.eye(1)  # penalty on control effort
    Q_du = 0 * np.eye(1)  # penalty on variation in control effort
    w = 1e4  # penalty on slack variables
    ts = 1  # time step that controller uses
    a_acc = 2.5  # acceleration limit
    a_dec = -2  # deceleration limit
    d_safe = 25
