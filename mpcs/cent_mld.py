import gurobipy as gp
import numpy as np
from dmpcpwa.mpc.mpc_mld_cent_decup import MpcMldCentDecup

from ACC_model import ACC


class MPCMldCent(MpcMldCentDecup):
    def __init__(
        self, systems: list[dict], acc: ACC, cost_2_norm: bool, n, N: int
    ) -> None:
        super().__init__(systems, n, N)
        self.n = n
        self.N = N
        self.setup_cost_and_constraints(self.u, acc, cost_2_norm)

    def setup_cost_and_constraints(self, u, acc, cost_2_norm):
        """Sets up the cost and constraints. Penalises the u passed in."""
        if cost_2_norm:
            cost_func = self.min_2_norm
        else:
            cost_func = self.min_1_norm

        nx_l = acc.nx_l
        nu_l = acc.nu_l
        Q_x_l = acc.Q_x_l
        Q_u_l = acc.Q_u_l
        Q_du_l = acc.Q_du_l
        sep = acc.sep
        d_safe = acc.d_safe
        w = acc.w

        # slack vars for soft constraints
        self.s = self.mpc_model.addMVar(
            (self.n, self.N + 1), lb=0, ub=float("inf"), name="s"
        )

        # formulate cost
        # leader_traj gets changed and fixed by setting its bounds
        self.leader_traj = self.mpc_model.addMVar(
            (nx_l, self.N + 1), lb=0, ub=0, name="leader_traj"
        )
        obj = 0
        for i in range(self.n):
            local_state = self.x[nx_l * i : nx_l * (i + 1), :]
            local_control = u[nu_l * i : nu_l * (i + 1), :]
            if i == 0:
                # first car follows traj with no sep
                follow_state = self.leader_traj
                temp_sep = np.zeros((2, 1))
            else:
                # otherwise follow car infront (i-1)
                follow_state = self.x[nx_l * (i - 1) : nx_l * (i), :]
                temp_sep = sep
            for k in range(self.N):
                obj += cost_func(
                    local_state[:, [k]] - follow_state[:, [k]] - temp_sep, Q_x_l
                )
                obj += cost_func(local_control[:, [k]], Q_u_l) + w * self.s[i, [k]]

                if k < self.N - 1:
                    obj += cost_func(
                        local_control[:, [k + 1]] - local_control[:, [k]], Q_du_l
                    )

            obj += (
                cost_func(
                    local_state[:, [self.N]] - follow_state[:, [self.N]] - temp_sep,
                    Q_x_l,
                )
                + w * self.s[i, [self.N]]
            )
        self.mpc_model.setObjective(obj, gp.GRB.MINIMIZE)

        # add extra constraints
        # acceleration constraints
        for i in range(self.n):
            for k in range(self.N):
                self.mpc_model.addConstr(
                    acc.a_dec * acc.ts
                    <= self.x[nx_l * i + 1, [k + 1]] - self.x[nx_l * i + 1, [k]],
                    name=f"dec_car_{i}_step{k}",
                )
                self.mpc_model.addConstr(
                    self.x[nx_l * i + 1, [k + 1]] - self.x[nx_l * i + 1, [k]]
                    <= acc.a_acc * acc.ts,
                    name=f"acc_car_{i}_step{k}",
                )

        # safe distance behind follower vehicle

        for i in range(self.n):
            local_state = self.x[nx_l * i : nx_l * (i + 1), :]
            if i != 0:  # leader isn't following another car
                follow_state = self.x[nx_l * (i - 1) : nx_l * (i), :]
                for k in range(self.N + 1):
                    self.mpc_model.addConstr(
                        local_state[0, [k]]
                        <= follow_state[0, [k]] - d_safe + self.s[i, [k]],
                        name=f"safe_dis_car_{i}_step{k}",
                    )

    def set_leader_traj(self, leader_traj):
        for k in range(self.N + 1):
            self.leader_traj[:, [k]].ub = leader_traj[:, [k]]
            self.leader_traj[:, [k]].lb = leader_traj[:, [k]]
