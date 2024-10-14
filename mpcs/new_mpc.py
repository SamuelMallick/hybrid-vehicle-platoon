from typing import Any, TypeVar

import casadi as cs
import numpy as np
from csnlp import Nlp
from csnlp.wrappers import Wrapper
from csnlp.wrappers.mpc.hybrid_mpc import HybridMpc

from misc.common_controller_params import Params

SymType = TypeVar("SymType", cs.SX, cs.MX)


class SolverTimeRecorder(Wrapper[SymType]):
    """A wrapper class of that records the time taken by the solver."""

    def __init__(self, *args: Any, **kwds: Any) -> None:
        super().__init__(*args, **kwds)
        self.solver_time: list[float] = []

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        sol = super().__call__(*args, **kwds)
        self.solver_time.append(sol.stats["t_wall_total"])
        return sol


class MpcMldCentNew(HybridMpc):
    """A centralized MPC controller for the platoon using mixed-integer MLD approach."""

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
        system: dict,
    ) -> None:
        super().__init__(nlp=Nlp[cs.SX](sym_type="SX"), prediction_horizon=N)
        self.n = 2
        self.m = 1
        self.N = N
        self.x, _ = self.state("x", self.n)
        self.u, _ = self.action("u", self.m)
        self.set_pwa_dynamics(system)
        self.setup_cost_and_constraints()

    def setup_cost_and_constraints(
        self,
    ):
        """Set up  cost and constraints for platoon tracking. Penalises the u passed in."""
        # slack vars for soft constraints

        # cost func
        # leader_traj - gets updated each time step
        self.leader_traj = self.parameter("leader_traj", (self.n, self.N + 1))
        self.fixed_parameters = {"leader_traj": np.zeros((self.n, self.N + 1))}
        cost = 0
        # tracking cost
        cost += sum(
            [
                (self.x[:, k] - self.leader_traj[:, k]).T
                @ self.Q_x
                @ (self.x[:, k] - self.leader_traj[:, k])
                for k in range(self.N + 1)
            ]
        )
        # control effort cost
        cost += sum([self.u[:, k].T @ self.Q_u @ self.u[:, k] for k in range(self.N)])
        # contral variation cost
        cost += sum(
            [
                (self.u[:, k + 1] - self.u[:, k]).T
                @ self.Q_du
                @ (self.u[:, k + 1] - self.u[:, k])
                for k in range(self.N - 1)
            ]
        )
        self.minimize(cost)

        # acceleration constraints
        self.constraint(
            "decel", self.a_dec * self.ts, "<=", self.x[1, 1:] - self.x[1, :-1]
        )
        self.constraint(
            "accel", self.x[1, 1:] - self.x[1, :-1], "<=", self.a_acc * self.ts
        )
        opts = {
            "expand": True,
            "show_eval_warnings": True,
            "warn_initial_bounds": True,
            "print_time": False,
            "record_time": True,
            "bound_consistency": True,
            "calc_lam_x": True,
            "calc_lam_p": False,
        }
        self.init_solver(opts, solver="bonmin")
