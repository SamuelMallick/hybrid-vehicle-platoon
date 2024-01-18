import gurobipy as gp
import numpy as np
from ACC_model import ACC

from dmpcpwa.mpc.mpc_mld import MpcMld


class MpcGear(MpcMld):
    """This is an MLD based MPC for a PWA system, that includes a discrete inputs which scale the input
    u. Inspired by the idea of gears, where u is scaled based on a choice of gear. The gear choices for
    each control signal are assumed decoupled."""

    def __init__(self, system: dict, N: int) -> None:
        """Init the MLD model and the MPC.
        Parameters
        ----------
        system: dict
            Dictionary containing the definition of the PWA system {S, R, T, A, B, c, D, E, F, G}.
             When S[i]x+R[x]u <= T[i] -> x+ = A[i]x + B[i]u + c[i].
             For MLD conversion the state and input must be constrained: Dx <= E, Fu <= G.
        N:int
            Prediction horizon length."""

        # init the PWA system as normal
        super().__init__(system, N)

    def setup_gears(self, N: int, acc: ACC, F: np.ndarray, G: np.ndarray):
        # now we constrain the input u to include the gears

        nu = self.u.shape[0]
        num_gears = len(acc.b)

        # new vars
        b = self.mpc_model.addMVar(
            (num_gears, nu, N), lb=-float("inf"), ub=float("inf"), name="b"
        )
        u_g = self.mpc_model.addMVar(
            (nu, N), lb=-float("inf"), ub=float("inf"), name="u_g"
        )
        sigma = self.mpc_model.addMVar(
            (num_gears, nu, N), vtype=gp.GRB.BINARY, name="sigma"
        )

        # constrain only one gear to be active at a time
        self.mpc_model.addConstrs(
            (
                gp.quicksum(sigma[j, i, k] for j in range(num_gears)) == 1
                for i in range(nu)
                for k in range(N)
            ),
            name="Sigma sum constraints",
        )

        # constraints along prediction horizon

        # first remove control constraints and readd them to u_g
        target_prefix = "control constraints"
        self.mpc_model.update()
        matching_constraints = [
            constr
            for constr in self.mpc_model.getConstrs()
            if constr.ConstrName.startswith(target_prefix)
        ]
        if len(matching_constraints) == 0:
            raise RuntimeError(
                "Couldn't get control constraint when creating gear MPC."
            )
        for constr in matching_constraints:
            self.mpc_model.remove(constr)

        for k in range(N):
            self.mpc_model.addConstr(
                F @ u_g[:, [k]] <= G,
                name="new control constraints",
            )

            for i in range(nu):  # each control signal
                # u_i = sum_j b_i,j
                self.mpc_model.addConstr(
                    self.u[i, [k]]
                    == gp.quicksum(b[j, i, k].reshape(1, 1) for j in range(num_gears)),
                )

                for j in range(num_gears):  # for each gear
                    M = acc.u_max * acc.b[j]
                    m = -acc.u_max * acc.b[j]
                    # the following four constraints make b_i,j = sigma_i,j * H[i] * u_g[i]
                    self.mpc_model.addConstr(b[j, i, [k]] <= M * sigma[j, i, [k]])
                    self.mpc_model.addConstr(b[j, i, [k]] >= m * sigma[j, i, [k]])

                    self.mpc_model.addConstr(
                        b[j, i, [k]]
                        <= acc.b[j] * u_g[i, [k]] - m * (1 - sigma[j, i, [k]])
                    )
                    self.mpc_model.addConstr(
                        b[j, i, [k]]
                        >= acc.b[j] * u_g[i, [k]] - M * (1 - sigma[j, i, [k]])
                    )

                    # next constraints force sigma to be active only when
                    # velocity conditions are satisfied.
                    M = acc.x2_max - acc.vh[j]
                    # pick out velocity associated with i'th control signal
                    f = self.x[2 * i + 1, [k]] - acc.vh[j]
                    self.mpc_model.addConstr(f <= M * (1 - sigma[j, i, [k]]))

                    M = acc.vl[j] - acc.x2_min
                    f = acc.vl[j] - self.x[2 * i + 1, [k]]
                    self.mpc_model.addConstr(f <= M * (1 - sigma[j, i, [k]]))

        self.u_g = u_g
        self.sigma = sigma
        self.b = b

    def solve_mpc(self, state):
        # override to return the actual input u_g
        u_0, info = super().solve_mpc(state)
        if self.mpc_model.Status == 2:  # check for successful solve
            u_g = self.u_g.X
            sig = self.sigma.X
            gears = np.ones((self.m, self.N))
            for i in range(self.m):
                for k in range(self.N):
                    gears[i, k] = sig[:, i, k].argmax() + 1
        else:
            u_g = np.zeros((self.m, self.N))
            gears = np.ones((self.m, self.N))  # default set all gears to one if infeas

        info["u"] = u_g
        self.gears_pred = gears
        return np.vstack((u_g[:, [0]], gears[:, [0]])), info
