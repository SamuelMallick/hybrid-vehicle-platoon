import gurobipy as gp
import numpy as np
from dmpcpwa.mpc.mpc_mld import MpcMld

from models import Vehicle


class MpcGear(MpcMld):
    """This is an MLD based MPC that uses discrete inputs that scale the input
    u. Inspired by the idea of gears, where u is scaled based on a choice of gear. The gear choices for
    each control signal are assumed decoupled.
    The dynamics are provided as a PWA system dict."""

    def __init__(self, system: dict, N: int, thread_limit: int | None = None, constrain_first_state: bool = False) -> None:
        """Init the MLD model of PWA system and the MPC wit hdiscrete inputs.
        Parameters
        ----------
        system: dict
            Dictionary containing the definition of the PWA system {S, R, T, A, B, c, D, E, F, G}.
             When S[i]x+R[x]u <= T[i] -> x+ = A[i]x + B[i]u + c[i].
             For MLD conversion the state and input must be constrained: Dx <= E, Fu <= G.
        N:int
            Prediction horizon length."""

        # init the PWA system - MLD constraints and binary variables for PWA dynamics
        super().__init__(
            system, N, thread_limit=thread_limit, constrain_first_state=constrain_first_state
        )

    def setup_gears(self, N: int, F: np.ndarray, G: np.ndarray):
        """Set up constraints in mixed-integer problem for gears."""
        nu = self.u.shape[0]
        num_gears = len(Vehicle.b)

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
        # first remove control constraints and re-add them to u_g
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
                    M: float = Vehicle.u_max * Vehicle.b[j]
                    m: float = Vehicle.u_min * Vehicle.b[j]
                    # the following four constraints make b_i,j = sigma_i,j * H[i] * u_g[i]
                    self.mpc_model.addConstr(b[j, i, [k]] <= M * sigma[j, i, [k]])
                    self.mpc_model.addConstr(b[j, i, [k]] >= m * sigma[j, i, [k]])

                    self.mpc_model.addConstr(
                        b[j, i, [k]]
                        <= Vehicle.b[j] * u_g[i, [k]] - m * (1 - sigma[j, i, [k]])
                    )
                    self.mpc_model.addConstr(
                        b[j, i, [k]]
                        >= Vehicle.b[j] * u_g[i, [k]] - M * (1 - sigma[j, i, [k]])
                    )

                    # next constraints force sigma to be active only when
                    # velocity conditions are satisfied.
                    M = Vehicle.v_max - Vehicle.vh[j]
                    # pick out velocity associated with i'th control signal
                    f = self.x[2 * i + 1, [k]] - Vehicle.vh[j]
                    self.mpc_model.addConstr(f <= M * (1 - sigma[j, i, [k]]))

                    M = Vehicle.vl[j] - Vehicle.v_min
                    f = Vehicle.vl[j] - self.x[2 * i + 1, [k]]
                    self.mpc_model.addConstr(f <= M * (1 - sigma[j, i, [k]]))

        self.u_g = u_g
        self.sigma = sigma
        self.b = b

    def solve_mpc(self, state, raises: bool = True):
        """Solve mpc for gear and throttle."""
        u_0, info = super().solve_mpc(state, raises=raises)
        if self.mpc_model.Status == 2:  # check for successful solve
            u_g = self.u_g.X
            sig = self.sigma.X
            gears = np.ones((self.m, self.N))
            for i in range(self.m):
                for k in range(self.N):
                    gears[i, k] = sig[:, i, k].argmax() + 1
        else:
            if raises:
                raise RuntimeWarning(f"gear mpc for state {state} is infeasible.")
            else:
                u_g = np.zeros((self.m, self.N))
                gears = 6* np.ones((self.m, self.N))  # default set all gears

        info["u"] = np.vstack((u_g, gears))
        self.gears_pred = gears
        return np.vstack((u_g[:, [0]], gears[:, [0]])), info
    
    def evaluate_cost(self, x0: np.ndarray, u: np.ndarray, j: np.ndarray | None = None):
        """Evalaute cost of MPC problem for a given x0 and u traj"""
        if u.shape != self.u.shape:
            raise ValueError(f'Expected u shape {self.u.shape}. Got {u.shape}.')
        if j is not None:
            for k_1 in range(u.shape[1]):   # loop horizon
                for k_2 in range(u.shape[0]):   # loop vehicles
                    for k_3 in range(6):    # loop gears
                        if j[k_2, k_1] == k_3 + 1:
                            self.sigma[k_3, k_2, k_1].ub = 1
                            self.sigma[k_3, k_2, k_1].lb = 1
                        else:
                            self.sigma[k_3, k_2, k_1].ub = 0
                            self.sigma[k_3, k_2, k_1].lb = 0
        
        self.IC.RHS = x0
        self.u_g.ub = u
        self.u_g.lb = u
        self.mpc_model.optimize()
        if self.mpc_model.Status == 2:  # check for successful solve
            cost = self.mpc_model.objVal
        else:
            cost = 'inf'
        self.x.ub = float('inf')
        self.x.lb = -float('inf')
        self.u_g.ub = float('inf')
        self.u_g.lb = -float('inf')
        if j is not None:
            for k_1 in range(u.shape[1]):   # loop horizon
                for k_2 in range(u.shape[0]):   # loop vehicles
                    for k_3 in range(6):    # loop gears
                            self.sigma[k_3, k_2, k_1].ub = float('inf')
                            self.sigma[k_3, k_2, k_1].lb = -float('inf')
        return cost
        

class MpcNonlinearGear(MpcGear):
    """An MPC controller than uses a nonlinear vehicle model along with discrete gear inputs x^+ = f(x) + B(j,x)u."""

    def __init__(
        self, systems: list[dict], N: int, thread_limit: int | None = None, constrain_first_state: bool = False
    ) -> None:
        """Instantiate mixed-integer model for nonlinear dynamics for len(systems) systems."""
        # build mixed-integer model
        mpc_model = gp.Model("non_linear_gear_mpc")
        mpc_model.setParam("OutputFlag", 0)
        mpc_model.setParam("Heuristics", 0)
        mpc_model.setParam("NonConvex", 2)
        if thread_limit is not None:
            mpc_model.params.threads = thread_limit

        # Uncomment if you need to differentiate between infeasbile and unbounded
        # mpc_model.setParam("DualReductions", 0)

        n = len(systems)
        nx_l = systems[0]["D"].shape[1]
        nu_l = systems[0]["F"].shape[1]

        # init states and control
        x = mpc_model.addMVar(
            (n * nx_l, N + 1), lb=-float("inf"), ub=float("inf"), name="x"
        )  # state
        u = mpc_model.addMVar(
            (n * nu_l, N), lb=-float("inf"), ub=float("inf"), name="u"
        )  # control

        # split the opt vars into local components
        x_l = [x[i * nx_l : (i + 1) * nx_l, :] for i in range(n)]
        u_l = [u[i * nu_l : (i + 1) * nu_l, :] for i in range(n)]

        # constraints for dynamics
        mpc_model.addConstrs(
            (
                x_l[i][:, [k + 1]] == systems[i]["dyn"](x_l[i][:, [k]], u_l[i][:, [k]])
                for i in range(n)
                for k in range(N)
            ),
            name="dynamics",
        )

        # control and state constraints
        mpc_model.addConstrs(
            (
                systems[i]["D"] @ x_l[i][:, [k]] <= systems[i]["E"]
                for i in range(n)
                for k in range(0 if constrain_first_state else 1, N + 1)
            ),
            name="state constraints",
        )
        mpc_model.addConstrs(
            (
                systems[i]["F"] @ u_l[i][:, [k]] <= systems[i]["G"]
                for i in range(n)
                for k in range(N)
            ),
            name="control constraints",
        )  # name is IMPORTANT - DONT CHANGE - see above MpcGear class
        # IC constraint - gets updated everytime solve_mpc is called
        self.IC = mpc_model.addConstr(x[:, [0]] == np.zeros((n * nx_l, 1)), name="IC")

        # assign parts of model to be used by class later
        self.mpc_model = mpc_model
        self.x = x
        self.u = u
        # CAREFUL - n here is number of agents, while m is dimension of global control var
        self.n = n
        self.m = n * nu_l
        self.N = N

    
