import gurobipy as gp
from dmpcpwa.mpc.mpc_mld_cent_decup import MpcMldCentDecup
from dmpcpwa.utils.pwa_models import cent_from_dist
from csnlp.wrappers.mpc.hybrid_mpc import HybridMpc
from csnlp import Nlp
from misc.common_controller_params import Params
from misc.spacing_policy import ConstantSpacingPolicy, SpacingPolicy
from models import Vehicle
import casadi as cs
import numpy as np


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
        o = {"gurobi": {"Heuristics": 1, "OptimalityTol": 1e-9}}
        self.init_solver(o, solver="gurobi")


class MpcMldCent(MpcMldCentDecup):
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
        n: int,
        N: int,
        pwa_systems: list[dict],
        spacing_policy: SpacingPolicy = ConstantSpacingPolicy(50),
        leader_index: int = 0,
        quadratic_cost: bool = True,
        thread_limit: int | None = None,
        accel_cnstr_tightening: float = 0.0,
        real_vehicle_as_reference: bool = False,
    ) -> None:
        super().__init__(
            pwa_systems, n, N, thread_limit=thread_limit, constrain_first_state=False
        )  # creates the state and control variables, sets the dynamics, and creates the MLD constraints for PWA dynamics
        self.n = n
        self.N = N

        self.setup_cost_and_constraints(
            self.u,
            spacing_policy,
            leader_index,
            quadratic_cost,
            accel_cnstr_tightening,
            real_vehicle_as_reference,
        )

    def setup_cost_and_constraints(
        self,
        u,
        spacing_policy: SpacingPolicy = ConstantSpacingPolicy(50),
        leader_index: int = 0,
        quadratic_cost: bool = True,
        accel_cnstr_tightening: float = 0.0,
        real_vehicle_as_reference: bool = False,
    ):
        """Set up  cost and constraints for platoon tracking. Penalises the u passed in."""
        if quadratic_cost:
            self.cost_func = self.min_2_norm
        else:
            self.cost_func = self.min_1_norm

        if leader_index != 0 and real_vehicle_as_reference:
            raise NotImplementedError(
                f"Not implemented for real vehicle with leader not 0."
            )

        nx_l = Vehicle.nx_l
        nu_l = Vehicle.nu_l

        # slack vars for soft constraints
        self.s = self.mpc_model.addMVar(
            (self.n, self.N + 1), lb=0, ub=float("inf"), name="s"
        )

        # cost func
        # leader_traj - gets updated each time step
        self.leader_traj = self.mpc_model.addMVar(
            (nx_l, self.N + 1), lb=0, ub=0, name="leader_traj"
        )
        cost = 0
        x_l = [self.x[i * nx_l : (i + 1) * nx_l, :] for i in range(self.n)]
        u_l = [u[i * nu_l : (i + 1) * nu_l, :] for i in range(self.n)]
        # tracking cost
        if not real_vehicle_as_reference:
            cost += sum(
                [
                    self.cost_func(
                        x_l[leader_index][:, [k]] - self.leader_traj[:, [k]], self.Q_x
                    )
                    for k in range(self.N + 1)
                ]
            )
        else:
            cost += sum(
                [
                    self.cost_func(
                        x_l[0][:, [k]]
                        - self.leader_traj[:, [k]]
                        - spacing_policy.spacing(x_l[0][:, [k]]),
                        self.Q_x,
                    )
                    for k in range(self.N + 1)
                ]
            )
        cost += sum(
            [
                self.cost_func(
                    x_l[i][:, [k]]
                    - x_l[i - 1][:, [k]]
                    - spacing_policy.spacing(x_l[i][:, [k]]),
                    self.Q_x,
                )
                for i in range(1, self.n)
                for k in range(self.N + 1)
            ]
        )
        # control effort cost
        cost += sum(
            [
                self.cost_func(u_l[i][:, [k]], self.Q_u)
                for i in range(self.n)
                for k in range(self.N)
            ]
        )
        # contral variation cost
        cost += sum(
            [
                self.cost_func(u_l[i][:, [k + 1]] - u_l[i][:, [k]], self.Q_du)
                for i in range(self.n)
                for k in range(self.N - 1)
            ]
        )
        # slack variable cost
        cost += sum(
            [self.w * self.s[i, k] for i in range(self.n) for k in range(self.N + 1)]
        )

        self.mpc_model.setObjective(cost, gp.GRB.MINIMIZE)

        # add extra constraints
        # acceleration constraints
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
        # safe distance behind follower vehicle
        if real_vehicle_as_reference:
            self.mpc_model.addConstrs(
                (
                    x_l[0][0, k] <= self.leader_traj[0, k] - self.d_safe + self.s[0, k]
                    for k in range(self.N + 1)
                ),
                name="safe_leader",
            )
        self.mpc_model.addConstrs(
            (
                x_l[i][0, k] <= x_l[i - 1][0, k] - self.d_safe + self.s[i, k]
                for i in range(1, self.n)
                for k in range(self.N + 1)
            ),
            name="safe",
        )

    def set_leader_traj(self, leader_traj):
        for k in range(self.N + 1):
            self.leader_traj[:, [k]].ub = leader_traj[:, [k]]
            self.leader_traj[:, [k]].lb = leader_traj[:, [k]]
