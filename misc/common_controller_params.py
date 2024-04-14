import numpy as np


class Params:
    Q_x = np.diag([1, 0.1])  # penalty of state tracking error
    Q_u = 1 * np.eye(1)  # penalty on control effort
    Q_du = 0 * np.eye(1)  # penalty on variation in control effort
    w = 1e4  # penalty on slack variables
    ts = 1  # time step that controller uses
    a_acc = 2.5  # acceleration limit
    a_dec = -2  # deceleration limit
    d_safe = 25
