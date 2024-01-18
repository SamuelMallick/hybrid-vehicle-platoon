import matplotlib.pyplot as plt
import numpy as np
from dmpcrl.utils.discretisation import forward_euler

# adaptive cruise control system. Single agent system outlined in:
# Adaptive Cruise Control for a SMART Car: A Comparison Benchmark for MPC-PWA Control Methods - D. Corona and B. De Schutter 2008


class ACC:
    nx_l = 2  # dimension of local state
    nu_l = 1  # dimension of local control
    ts = 1  # time step size for discretisation

    # local costs
    Q_x_l = np.diag([1, 0.1])
    Q_u_l = 1 * np.eye(nu_l)  # penalty on control effort
    Q_du_l = 0 * np.eye(nu_l)  # penalty on variation in control effort
    w = 1e4  # penalty on slack violations

    sep = np.array([[-50], [0]])  # desired seperation between vehicles states

    mass = 800  # mass
    m_inhom = [
        np.random.uniform(725, 1043) for i in range(20)
    ]  # mass value for inhomogenous platoon
    c_fric = 0.5  # viscous friction coefficient
    mu = 0.01  # coulomb friction coefficient
    grav = 9.8  # gravity accel
    w_min = 105  # min rot speed rad/s
    w_max = 630  # max rot speed rad/s

    x1_min = 0  # min pos
    x1_max = 20000  # max_pos
    x2_min = 3.94  # min velocity
    x2_max = 45.84  # max velocity
    u_max = 1  # max throttle/brake
    a_acc = 2.5  # comfort acc
    a_dec = -2  # comfort dec
    d_safe = 25  # safe pos

    # transmission rate for each of the 6 gears
    p = [14.203, 10.310, 7.407, 5.625, 4.083, 2.933]
    b = [4057, 2945, 2116, 1607, 1166, 838]  # max traction force for each gear
    vl = [3.94, 5.43, 7.56, 9.96, 13.70, 19.10]
    vh = [9.46, 13.04, 18.15, 23.90, 32.93, 45.84]
    Te_max = 80  # maximum engine torque - constant in the range 200 < w < 480

    # PWA approximation of friction c*x2^2 = c1*x2 if x2 <= x2_max/2, = c2*x2-d
    beta = (3 * c_fric * x2_max**2) / (16)
    alpha = x2_max / 2
    c1 = beta / alpha
    c2 = (c_fric * x2_max**2 - beta) / (x2_max - alpha)
    d = beta - alpha * ((c_fric * x2_max**2 - beta) / (x2_max - alpha))
    # d = 0.230769

    # PWA regions velocity upper limits for gear switches
    v_gear_lim = []
    for i in range(1, 6):
        v_gear_lim.append((vh[i] - vl[i]) / 2 + vl[i])

    def build_full_pwa_system(self, mass):
        # build full PWA system
        s = 7  # 7 PWA regions
        r = 2  # number of rows in Sx + RU <= T conditions
        S = []
        R = []
        T = []
        A = []
        B = []
        c = []

        for i in range(s):
            S.append(np.array([[0, 1], [0, -1]]))
            R.append(np.zeros((r, 1)))

        # manually append the limits
        T.append(np.array([[self.v_gear_lim[0]], [-self.x2_min]]))
        T.append(np.array([[self.v_gear_lim[1]], [-self.v_gear_lim[0]]]))
        T.append(np.array([[self.v_gear_lim[2]], [-self.v_gear_lim[1]]]))
        T.append(np.array([[self.alpha], [-self.v_gear_lim[2]]]))
        T.append(np.array([[self.v_gear_lim[3]], [-self.alpha]]))
        T.append(np.array([[self.v_gear_lim[4]], [-self.v_gear_lim[3]]]))
        T.append(np.array([[self.x2_max], [-self.v_gear_lim[4]]]))

        # manually append the A matrices - first three regions have c1 and last four have c2 for friction
        A.append(np.array([[0, 1], [0, -(self.c1) / (mass)]]))
        A.append(np.array([[0, 1], [0, -(self.c1) / (mass)]]))
        A.append(np.array([[0, 1], [0, -(self.c1) / (mass)]]))
        A.append(np.array([[0, 1], [0, -(self.c1) / (mass)]]))
        A.append(np.array([[0, 1], [0, -(self.c2) / (mass)]]))
        A.append(np.array([[0, 1], [0, -(self.c2) / (mass)]]))
        A.append(np.array([[0, 1], [0, -(self.c2) / (mass)]]))

        # manually append B matrices
        B.append(np.array([[0], [(self.b[0]) / (mass)]]))
        B.append(np.array([[0], [(self.b[1]) / (mass)]]))
        B.append(np.array([[0], [(self.b[2]) / (mass)]]))
        # fourth and fifth share same gear as the split is over the friction coeff
        B.append(np.array([[0], [(self.b[3]) / (mass)]]))
        B.append(np.array([[0], [(self.b[3]) / (mass)]]))
        B.append(np.array([[0], [(self.b[4]) / (mass)]]))
        B.append(np.array([[0], [(self.b[5]) / (mass)]]))

        # manually append c matrices - last four regions have offset d due to friction PWA
        c.append(np.array([[0], [-self.mu * self.grav]]))
        c.append(np.array([[0], [-self.mu * self.grav]]))
        c.append(np.array([[0], [-self.mu * self.grav]]))
        c.append(np.array([[0], [-self.mu * self.grav]]))
        c.append(np.array([[0], [-self.mu * self.grav - self.d / mass]]))
        c.append(np.array([[0], [-self.mu * self.grav - self.d / mass]]))
        c.append(np.array([[0], [-self.mu * self.grav - self.d / mass]]))

        # discretise the dynamics
        Ad = []
        Bd = []
        cd = []
        for i in range(s):
            Ad_i, Bd_i, cd_i = forward_euler(A[i], B[i], self.ts, c[i])
            Ad.append(Ad_i)
            Bd.append(Bd_i)
            cd.append(cd_i)

        D = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        E = np.array([[self.x1_max], [-self.x1_min], [self.x2_max], [-self.x2_min]])
        F = np.array([[1], [-1]])
        G = np.array([[self.u_max], [self.u_max]])

        return {
            "S": S,
            "R": R,
            "T": T,
            "A": Ad,
            "B": Bd,
            "c": cd,
            "D": D,
            "E": E,
            "F": F,
            "G": G,
        }

    def build_friction_pwa_system(self, mass):
        # build smaller PWA system for just the friction, which is used with dicrete input model of gears
        s = 2  # 7 PWA regions
        r = 2  # number of rows in Sx + RU <= T conditions
        S = []
        R = []
        T = []
        A = []
        B = []
        c = []

        for i in range(s):
            S.append(np.array([[0, 1], [0, -1]]))
            R.append(np.zeros((r, 1)))

        T.append(np.array([[self.alpha], [-self.x2_min]]))
        T.append(np.array([[self.x2_max], [-self.alpha]]))

        A.append(np.array([[0, 1], [0, -(self.c1) / (mass)]]))
        A.append(np.array([[0, 1], [0, -(self.c2) / (mass)]]))

        B.append(np.array([[0], [(1) / (mass)]]))
        B.append(np.array([[0], [(1) / (mass)]]))

        c.append(np.array([[0], [-self.mu * self.grav]]))
        c.append(np.array([[0], [-self.mu * self.grav - self.d / mass]]))

        # discretise the dynamics
        Ad = []
        Bd = []
        cd = []
        for i in range(s):
            Ad_i, Bd_i, cd_i = forward_euler(A[i], B[i], self.ts, c[i])
            Ad.append(Ad_i)
            Bd.append(Bd_i)
            cd.append(cd_i)

        D = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        E = np.array([[self.x1_max], [-self.x1_min], [self.x2_max], [-self.x2_min]])
        F = np.array([[1], [-1]])
        G = np.array([[self.u_max], [self.u_max]])

        return {
            "S": S,
            "R": R,
            "T": T,
            "A": Ad,
            "B": Bd,
            "c": cd,
            "D": D,
            "E": E,
            "F": F,
            "G": G,
        }

    def leader_state_1(self, ep_len, N):
        """Leader trajectory with constant velocity"""
        leader_state = np.zeros((2, ep_len + N + 1))
        leader_speed = 20
        leader_initial_pos = 3000
        leader_state[:, [0]] = np.array([[leader_initial_pos], [leader_speed]])
        for k in range(ep_len + N):
            leader_state[:, [k + 1]] = leader_state[:, [k]] + self.ts * np.array(
                [[leader_speed], [0]]
            )
        return leader_state

    def leader_state_2(self, ep_len, N):
        """Leader trajectory with speed up and slow down to same initial speed."""
        leader_state = np.zeros((2, ep_len + N + 1))
        leader_speed = 20
        leader_initial_pos = 600
        leader_state[:, [0]] = np.array([[leader_initial_pos], [leader_speed]])
        for k in range(int(ep_len / 4)):
            leader_state[:, [k + 1]] = np.array(
                [[leader_state[0, k]], [leader_speed]]
            ) + self.ts * np.array([[leader_speed], [0]])
        leader_speed = 30
        for k in range(int(ep_len / 4), int(1 * ep_len / 2)):
            leader_state[:, [k + 1]] = np.array(
                [[leader_state[0, k]], [leader_speed]]
            ) + self.ts * np.array([[leader_speed], [0]])
        leader_speed = 20
        for k in range(int(1 * ep_len / 2), ep_len + N):
            leader_state[:, [k + 1]] = np.array(
                [[leader_state[0, k]], [leader_speed]]
            ) + self.ts * np.array([[leader_speed], [0]])
        return leader_state

    def leader_state_3(self, ep_len, N):
        """Leader trajectory with speed up, then slow down to slower than initial, then speed back up to initial speed."""
        leader_state = np.zeros((2, ep_len + N + 1))
        leader_speed = 20
        leader_initial_pos = 600
        leader_state[:, [0]] = np.array([[leader_initial_pos], [leader_speed]])
        for k in range(int(ep_len / 4)):
            leader_state[:, [k + 1]] = np.array(
                [[leader_state[0, k]], [leader_speed]]
            ) + self.ts * np.array([[leader_speed], [0]])
        leader_speed = 30
        for k in range(int(ep_len / 4), int(1 * ep_len / 2)):
            leader_state[:, [k + 1]] = np.array(
                [[leader_state[0, k]], [leader_speed]]
            ) + self.ts * np.array([[leader_speed], [0]])
        leader_speed = 10
        for k in range(int(1 * ep_len / 2), int(3 * ep_len / 4)):
            leader_state[:, [k + 1]] = np.array(
                [[leader_state[0, k]], [leader_speed]]
            ) + self.ts * np.array([[leader_speed], [0]])
        leader_speed = 20
        for k in range(int(3 * ep_len / 4), ep_len + N):
            leader_state[:, [k + 1]] = np.array(
                [[leader_state[0, k]], [leader_speed]]
            ) + self.ts * np.array([[leader_speed], [0]])
        return leader_state

    def __init__(self, ep_len, N, leader_traj=1):
        if leader_traj == 1:
            self.leader_state = self.leader_state_1(ep_len, N)
        elif leader_traj == 2:
            self.leader_state = self.leader_state_2(ep_len, N)
        elif leader_traj == 3:
            self.leader_state = self.leader_state_3(ep_len, N)
        else:
            raise RuntimeError(f"No leader traj asosciated with integer {leader_traj}")

    def get_pwa_gear_from_speed(self, v):
        """Get the gear j from the speed v, by the PWA model."""
        # check gear 2 to 5
        for i in range(len(self.b) - 2):
            if v >= self.v_gear_lim[i] and v < self.v_gear_lim[i + 1]:
                return i + 2

        # check gear 1
        if v < self.v_gear_lim[0] and v >= self.x2_min:
            return 1
        if v >= self.v_gear_lim[-1] and v <= self.x2_max:
            return 6
        raise RuntimeError(f"Didn't find any gear for the given speed {v}")

    def get_traction_from_gear(self, j):
        """Get the corresponding constant traction force for the gear j."""
        if j < 1 or j > 6:
            raise RuntimeError("Gear value out of range.")
        if j % 1 != 0:
            raise RuntimeError("Gear value is not an int.")
        gear = int(j)
        return self.b[gear - 1]

    def get_pwa_system(self, index=None):
        """Get the full pwa system dictionary."""
        if index is None:
            return self.build_full_pwa_system(self.mass)
        return self.build_full_pwa_system(self.m_inhom[index])

    def get_friction_pwa_system(self, index=None):
        """Get the friction pwa system dictionary."""
        if index is None:
            return self.build_friction_pwa_system(self.mass)
        return self.build_friction_pwa_system(self.m_inhom[index])

    # the true non-linear dynamics of the car
    def step_car_dynamics_nl(self, x, u, j, n, ts, homog=True):
        """Steps the car dynamics for n cars with non-linear model by ts seconds. x is state, u is control, j is gears."""
        num_steps = 10
        DT = ts / num_steps
        for t in range(num_steps):
            x_temp = np.zeros(x.shape)
            for i in range(n):
                if homog:
                    mass = self.mass
                else:
                    mass = self.m_inhom[i]
                x_l = x[self.nx_l * i : self.nx_l * (i + 1), :]  # get local state
                u_l = u[self.nu_l * i : self.nu_l * (i + 1), :]  # get local control
                j_l = j[self.nu_l * i : self.nu_l * (i + 1), :]  # get local gear
                f = np.array(
                    [
                        [x_l[1, 0]],
                        [
                            -(self.c_fric * x_l[1, 0] ** 2) / (mass)
                            - self.mu * self.grav
                        ],
                    ]
                )
                B = np.array([[0], [self.get_traction_from_gear(j_l) / mass]])
                x_temp[self.nx_l * i : self.nx_l * (i + 1), :] = x_l + DT * (
                    f + B * u_l
                )

                # TODO handle this better
                # force velocity to be above 2 where PWA dynamics are valid
                if x_temp[self.nx_l * (i) + 1, :] < self.x2_min:
                    x_temp[self.nx_l * (i) + 1, :] = self.x2_min

            x = x_temp

        return x

    def step_car_dynamics_pwa(self, x, u, n, ts):
        """Steps the car dynamics for n cars with pwa model by ts seconds. x is state, u is control."""
        x_temp = np.zeros(x.shape)
        for i in range(n):
            x_l = x[self.nx_l * i : self.nx_l * (i + 1), :]  # get local state
            u_l = u[self.nu_l * i : self.nu_l * (i + 1), :]  # get local control
            for j in range(len(self.pwa_system["S"])):
                if all(
                    self.pwa_system["S"][j] @ x_l + self.pwa_system["R"][j] @ u_l
                    <= self.pwa_system["T"][j]
                    + np.array(
                        [[0], [1e-4]]
                    )  # buffer is to have one of the as a strict inequality
                ):
                    x_pwa = (
                        self.pwa_system["A"][j] @ x_l
                        + self.pwa_system["B"][j] @ u_l
                        + self.pwa_system["c"][j]
                    )
                    break
            x_temp[self.nx_l * i : self.nx_l * (i + 1), :] = x_pwa

        return x_temp

    def get_u_for_constant_vel(self, v, index=None):
        """returns the control input which will keep the velocity v constant, as by the PWA dynamics."""
        x = np.array([[0], [v[0]]])  # first state does not matter for this pwa sys
        u = np.array([[0]])  # neither does control

        pwa_system = self.get_pwa_system(index)

        for j in range(len(pwa_system["S"])):
            if all(
                pwa_system["S"][j] @ x + pwa_system["R"][j] @ u
                <= pwa_system["T"][j]
                + np.array(
                    [[0], [1e-4]]
                )  # buffer is to have one of the as a strict inequality
            ):
                # This is VERY specific to this system, DO NOT reuse this code on other PWA systems.
                return (1 / pwa_system["B"][j][1, 0]) * (
                    v - pwa_system["A"][j][1, 1] * v - pwa_system["c"][j][1, 0]
                )

        raise RuntimeError("Didn't find any PWa region for the given speed!")

    def get_leader_state(self):
        return self.leader_state


if __name__ == "__main__":
    # check validity of PWA approximation

    acc = ACC(0, 0)
    pwa_sys = acc.get_pwa_system()

    test_len = 100
    np.random.seed(0)
    x0 = np.array([[10], [5], [20], [6]])
    u = np.random.random((2, test_len))  # random control samples between 0 and 1
    x_nl = np.zeros((4, test_len))  # non-linear traj
    x_pwa = np.zeros((4, test_len))  # pwa traj

    # set IC
    x_nl[:, [0]] = x0
    x_pwa[:, [0]] = x0
    for t in range(test_len - 1):
        # step non linear
        x_nl[:, [t + 1]] = acc.step_car_dynamics_nl(x_nl[:, [t]], u[:, [t]], 2, acc.ts)
        x_pwa[:, [t + 1]] = acc.step_car_dynamics_pwa(
            x_nl[:, [t]], u[:, [t]], 2, acc.ts
        )

    _, axs = plt.subplots(2, 1, constrained_layout=True, sharex=True)
    axs[0].plot(x_nl[0, :], "r")
    axs[0].plot(x_pwa[0, :], "b")
    axs[1].plot(x_nl[1, :], "r")
    axs[1].plot(x_pwa[1, :], "b")
    plt.show()
