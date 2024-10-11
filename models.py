import warnings
from typing import Literal

import gurobipy as gp
import matplotlib.pyplot as plt
import numpy as np
from dmpcrl.utils.discretisation import forward_euler


class GearTransimission:
    """Class for modelling gear traction dynamics in Adaptive Cruise Control for a SMART Car: A Comparison Benchmark for MPC-PWA Control Methods - D. Corona and B. De Schutter 2008."""

    t = [
        [253.54, 4056.7, 3042, 52],
        [184, 2944.75, 2208.55],
        [132.22, 2115.6, 1586.7],
        [100 / 415, 1605, 1205],
        [72.88, 1166, 874.7],
        [52.4, 838, 628.3],
    ]  # 3 values defining traction in traction curve for each gear. See paper.
    v = [
        [2.0706, 4.12158, 9.29, 12.38],
        [2.85, 5.675, 12.7956, 17.06],
        [3.9705, 7.90316, 17.8105, 23.7474],
        [5.228, 10.42, 23.454, 31.2704],
        [7.203, 14.335, 32.31, 43.0802],
        [10.027, 19.956, 44.978, 59.9715],
    ]  # 4 values defining velocity in traction curve for each gear. See paper.

    def get_traction(self, v: float, j: int):
        """Get trajection value for speed v ms^-1 and gear j."""
        if j < 1 or j > 6:
            raise RuntimeError(f"Gear value out of range {1} - {6}.")
        if v < 2.0706 or v > 59.9715:
            raise RuntimeError(f"Velocity value out of range {2.0706} - {59.9715}.")

        j = j - 1  # shift index as list start from 0

        if v <= self.v[j][0] or v >= self.v[j][3]:
            raise RuntimeError(
                f"Velocity {v} out of range {self.v[j][0]} - {self.v[j][3]} for gear {j + 1}."
            )
        if v < self.v[j][1]:  # rising part of traction curve
            return ((v - self.v[j][0]) / (self.v[j][1] - self.v[j][0])) * (
                self.t[j][1] - self.t[j][0]
            ) + self.t[j][0]
        elif v > self.v[j][2]:  # falling part of traction curve
            return self.t[j][1] - (
                (v - self.v[j][2]) / (self.v[j][3] - self.v[j][2])
            ) * (self.t[j][1] - self.t[j][2])
        return self.t[j][1]  # flat part of traction curve


class Vehicle:
    """Class for vehicle with true nonlinear and hybrid vehicle dynamics."""

    nx_l = 2  # dimension of local state
    nu_l = 1  # dimension of local control

    # model coefficients
    c_fric = 0.5  # viscous friction coefficient
    mu = 0.01  # coulomb friction coefficient
    grav = 9.8  # gravity accel
    w_min = 105  # min rot speed rad/s
    w_max = 630  # max rot speed rad/s
    p = [
        14.203,
        10.310,
        7.407,
        5.625,
        4.083,
        2.933,
    ]  # transmission rate for each of the 6 gears
    b = [4057, 2945, 2116, 1607, 1166, 838]  # max traction force for each gear

    # upper and lower velocity range for gears
    vl = [
        3.94,
        5.43,
        7.56,
        9.96,
        13.70,
        19.10,
    ]
    vh = [9.46, 13.04, 18.15, 23.90, 32.93, 45.84]
    v_min, v_max = vl[0], vh[-1]  # velocity limits based on gear ranges
    u_min, u_max = -1.0, 1.0  # throttle input limits
    p_min, p_max = (
        0.0,
        10000.0,
    )  # abritrary position bounds in PWA models (conversion to PWA requires bounded states)
    Te_max = 80  # maximum engine torque - constant in the range 200 < w < 480

    def __init__(self, m: int = 800) -> None:
        """Create vehicle with mass m."""
        self.m = m
        self.gear_model = GearTransimission()

    def A(self, x: np.ndarray):
        """A(x) component of non-linear hybrid dynamics."""
        v = x[1, 0]  # velocity component of state
        return np.array(
            [
                [v],
                [-(self.c_fric * v**2) / (self.m) - self.mu * self.grav],
            ]
        )

    def B(self, x: np.ndarray, j: int):
        """B(j, x) component of non-linear hybrid dynamics."""
        v = x[1, 0]  # velocity component of state
        return np.array([[0], [self.gear_model.get_traction(v, j) / self.m]])

    def step(self, x: np.ndarray, u: float, j: int, ts: float):
        """Step local non-linear hybrid dynamics with euler step of ts seconds."""
        if np.abs(u) > 1 + 1e5:  # small numerical tolerance for control bound
            raise ValueError("Control u is bounded -1 <= u <= 1.")

        if x[1, 0] < self.gear_model.v[0][0] or x[1, 0] > self.gear_model.v[-1][-1]:
            raise RuntimeError(
                f"Velocity {x[1, 0]} of vehicle exeeds true model bounds {self.gear_model.v[0][0], self.gear_model.v[-1][-1]}."
            )

        x_new = x + ts * (self.A(x) + self.B(x, j) * u)
        return x_new

    def dyn(self, x, u, ts):
        """Nonlinear discrete dynamics for gurobipy sybolic states and actions, with euler time step ts."""
        if x.shape != (2, 1) or u.shape != (1, 1):
            raise ValueError(
                f"Dyn function for nonlinear model only defined for x and u of size {(2, 1)} and {(1,1)}."
            )
        A = self.A(x)
        B = np.array([[0], [(1) / (self.m)]]) @ u
        o = gp.MQuadExpr.zeros((2, 1))
        o[0, 0] = A[0, 0].item() + B[0, 0]
        o[1, 0] = A[1, 0].item() + B[1, 0]
        return x + ts * o

    def get_discrete_system(self, ts):
        """Return system dictionary for discrete dynamics with time step ts.
        Dict is discription of nonlinear model. In this model the u contains the traction force of the gear already.
        o['dyn'] is lambda function for dynamics. It is SPECIFICALLY constructed for Gurobipy objects. This is not compatible with other forms yet.
        o['D'], o['E'], o['F'], o['G'] are state and control constraints Dx <= E, F@u <= G
        """
        D = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        E = np.array([[self.p_max], [-self.p_min], [self.v_max], [-self.v_min]])
        F = np.array([[1], [-1]])
        G = np.array(
            [[self.u_max], [-self.u_min]]
        )  # velocity bounded -1 <= u <= 1 as it is normalized throttle

        return {
            "dyn": lambda x, u: self.dyn(x, u, ts),
            "D": D,
            "E": E,
            "F": F,
            "G": G,
        }

    def get_gear_from_velocity(self, v: float):
        """Get a gear j that is valid for the velocity v."""
        if v < self.v_min or v > self.v_max:
            warnings.warn(
                f"Velocity {v} is not within bounds {self.v_min}/{self.v_max}"
            )
            if v < self.v_min:  # return first gear if velocity is below min
                return 1
            if v > self.v_max:  # last gear if above max
                return 6
        for i in range(len(self.b)):
            if v > self.vl[i] and v < self.vh[i]:  # return first valid gear found
                return i + 1
        raise ValueError(f"No gear found for velocity {v}")

    def get_u_for_constant_vel(self, v: float, j: int):
        """Get the control input which will keep the velocity v constant with gear j, as by the nonlinear cont time dynamics."""
        x = np.array([[0], [v]])  # first state does not matter for this pwa sys
        u = np.array([[0]])  # neither does control

        if j < 1 or j > 6:
            raise ValueError(f"{j} is not a valid gear.")

        j = j - 1
        if v < self.v_min and j == 0:
            warnings.warn(
                f"Velocity {v} is below min {self.v_min}, using first gear but result will be approximate."
            )
        elif v > self.v_max and j == 5:
            warnings.warn(
                f"Velocity {v} is above max {self.v_max}, using last gear but result will be approximate."
            )
        elif v < self.vl[j] or v > self.vh[j]:
            raise ValueError(f"Velocity {v} is not valid for gear {j+1}")

        return (self.c_fric * v * v + self.mu * self.m * self.grav) / (self.b[j])


class Platoon:
    """Class for platoon of vehicles."""

    nx_l = Vehicle.nx_l  # dimension of local state for each vehicle
    nu_l = Vehicle.nu_l  # dimension of local action for each vehicle

    def __init__(
        self,
        n: int,
        vehicle_type: Literal["nonlinear", "pwa_friction", "pwa_gear"],
        masses: list[None] | None = None,
    ) -> None:
        """Create platoon with n vehicles vehicles i having mass masses[i]. If masses is None all vehicles have default mass."""
        if vehicle_type == "nonlinear":
            vehicle_class = Vehicle
        elif vehicle_type == "pwa_friction":
            vehicle_class = PwaFrictionVehicle
        elif vehicle_type == "pwa_gear":
            vehicle_class = PwaGearVehicle
        else:
            raise ValueError(f"{vehicle_type} is not a valid vehicle type.")

        self.n = n  # number of vehicles in platoon
        self.vehicles: list[Vehicle] = []
        if masses is not None:
            if len(masses) != n:
                raise ValueError(f"Required {n} vehicles masses. Got {len(masses)}.")

            for i in range(n):
                self.vehicles.append(vehicle_class(m=masses[i]))
        else:
            for i in range(n):
                self.vehicles.append(vehicle_class())

    def get_vehicles(self):
        return self.vehicles

    def step_platoon(self, x: np.ndarray, u: np.ndarray, j: np.ndarray, ts: float):
        """Steps the platoon with non-linear model by ts seconds. x is state, u is control, j is gears."""
        if (
            x.shape != (self.nx_l * self.n, 1)
            or u.shape != (self.nu_l * self.n, 1)
            or j.shape != (self.n, 1)
        ):
            raise ValueError(f"Dimension error in x, u, or j.")

        num_steps = 10  # number of sub-steps of nonlinear model within ts
        DT = ts / num_steps
        for _ in range(num_steps):
            # split global vars into list of local
            x_l = np.split(x, self.n, axis=0)
            u_l = np.split(u, self.n, axis=0)
            j_l = np.split(j, self.n, axis=0)
            x_new = [
                self.vehicles[i].step(x_l[i], u_l[i].item(), int(j_l[i].item()), DT)
                for i in range(self.n)
            ]
            x = np.vstack(x_new)
        return x

    def get_gear_from_vehicle_velocity(self, i: int, v: float):
        """Get a gear, given the velocity of a given vehicle in the platoon."""
        if not isinstance(self.vehicles[i], PwaGearVehicle):
            raise RuntimeError(
                f"Gear from velocity asked but the given vehicle {i} is not a PWA vehicle."
            )
        return self.vehicles[i].get_gear_from_velocity(v)

    def get_vehicle_system_dicts(self, ts: float) -> list[dict]:
        """Get the dictionary representation of the PWA system for each vehicle."""
        return [vehicle.get_discrete_system(ts) for vehicle in self.vehicles]


class PwaFrictionVehicle(Vehicle):
    """A vehicle with PWA approximation replacing nonlinear friction."""

    # PWA friction description
    beta = (3 * Vehicle.c_fric * Vehicle.v_max**2) / (16)
    alpha = Vehicle.v_max / 2
    c1 = beta / alpha
    c2 = (Vehicle.c_fric * Vehicle.v_max**2 - beta) / (Vehicle.v_max - alpha)
    d = beta - alpha * (
        (Vehicle.c_fric * Vehicle.v_max**2 - beta) / (Vehicle.v_max - alpha)
    )

    def __init__(self, m: int = 800) -> None:
        Vehicle.__init__(self, m)
        self.system = self.build_friction_pwa_system(m)

    def build_friction_pwa_system(self, mass: int, bound_velocity: bool = False):
        """Build PWA approximation of friction c*x2^2 = c1*x2 if x2 <= x2_max/2, = c2*x2-d."""
        # build PWA system representation in dictionary
        if bound_velocity:
            S = [np.array([[0, 1], [0, -1]]), np.array([[0, 1], [0, -1]])]
            T = [
                np.array([[self.alpha], [-self.v_min]]),
                np.array([[self.v_max], [-self.alpha]]),
            ]  # i.e., v >= v_min is part of PWA constraint
        else:
            S = [np.array([[0, 1], [0, 0]]), np.array([[0, 0], [0, -1]])]
            T = [
                np.array([[self.alpha], [0]]),
                np.array([[0], [-self.alpha]]),
            ]
        R = [np.zeros((2, 1)), np.zeros((2, 1))]

        A = [
            np.array([[0, 1], [0, -(self.c1) / (mass)]]),
            np.array([[0, 1], [0, -(self.c2) / (mass)]]),
        ]
        B = [np.array([[0], [(1) / (mass)]]), np.array([[0], [(1) / (mass)]])]
        c = [
            np.array([[0], [-self.mu * self.grav]]),
            np.array([[0], [-self.mu * self.grav - self.d / mass]]),
        ]
        D = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        E = np.array([[self.p_max], [-self.p_min], [self.v_max], [-self.v_min]])
        F = np.array([[1], [-1]])
        G = np.array(
            [[self.u_max], [-self.u_min]]
        )  # velocity bounded -1 <= u <= 1 as it is normalized throttle

        return {
            "S": S,
            "R": R,
            "T": T,
            "A": A,
            "B": B,
            "c": c,
            "D": D,
            "E": E,
            "F": F,
            "G": G,
        }

    def get_u_for_constant_vel(self, v: float, j: int):
        """Get the control input which will keep the velocity v constant with gear j, as by the PWA dynamics."""
        x = np.array([[0], [v]])  # first state does not matter for this pwa sys
        u = np.array([[0]])  # neither does control

        if j < 1 or j > 6:
            raise ValueError(f"{j} is not a valid gear.")

        j = j - 1
        if v < self.v_min and j == 0:
            warnings.warn(
                f"Velocity {v} is below min {self.v_min}, using first gear but result will be approximate."
            )
        elif v > self.v_max and j == 5:
            warnings.warn(
                f"Velocity {v} is above max {self.v_max}, using last gear but result will be approximate."
            )
        elif v < self.vl[j] or v > self.vh[j]:
            raise ValueError(f"Velocity {v} is not valid for gear {j+1}")

        for i in range(len(self.system["S"])):
            if all(
                self.system["S"][i] @ x + self.system["R"][i] @ u
                <= self.system["T"][i]
                + np.array(
                    [[0], [1e-4]]
                )  # buffer is to have one of the as a strict inequality
            ):
                # This is VERY specific to this system, DO NOT reuse this code on other PWA systems.
                # we are setting the \dot{v} = 0, and solving for the input - using cont time model
                return (1 / (self.b[j] * self.system["B"][i][1, 0])) * (
                    -self.system["A"][i][1, 1] * v - self.system["c"][i][1, 0]
                )

        raise RuntimeError("Didn't find any PWA region for the given speed!")

    def get_discrete_system(self, ts: float):
        """Return PWA dictionary for discrete dynamics with time step ts."""
        disc_PWA_sys = self.system.copy()
        # discretise the dynamics
        Ad = []
        Bd = []
        cd = []
        for i in range(len(disc_PWA_sys["A"])):
            Ad_i, Bd_i, cd_i = forward_euler(
                disc_PWA_sys["A"][i], disc_PWA_sys["B"][i], ts, disc_PWA_sys["c"][i]
            )
            Ad.append(Ad_i)
            Bd.append(Bd_i)
            cd.append(cd_i)
        disc_PWA_sys["A"] = Ad
        disc_PWA_sys["B"] = Bd
        disc_PWA_sys["c"] = cd
        return disc_PWA_sys


class PwaGearVehicle(PwaFrictionVehicle):
    """A Vehicle with PWA approximation for gears and nonlinear friction."""

    def __init__(self, m: int = 800) -> None:
        Vehicle.__init__(self, m)
        self.system = self.build_gear_pwa_system(m)

    def build_gear_pwa_system(self, mass: int, bound_velocity: bool = False):
        """Build PWA approximation of gears and frition."""

        # PWA regions velocity upper limits for gear switches
        self.v_gear_lim = []
        for i in range(1, 6):
            self.v_gear_lim.append((self.vh[i] - self.vl[i]) / 2 + self.vl[i])

        s = 7  # 7 PWA regions
        r = 2  # number of rows in Sx + RU <= T conditions
        S = []
        R = []
        T = []
        A = []
        B = []
        c = []

        S.append(
            np.array([[0, 1], [0, -1]])
            if bound_velocity
            else np.array([[0, 1], [0, 0]])
        )
        for i in range(1, s - 1):
            S.append(np.array([[0, 1], [0, -1]]))
        S.append(
            np.array([[0, 1], [0, -1]])
            if bound_velocity
            else np.array([[0, 0], [0, -1]])
        )

        R = [np.zeros((r, 1)) for _ in range(s)]

        # manually append the limits
        T.append(
            np.array([[self.v_gear_lim[0]], [-self.v_min]])
            if bound_velocity
            else np.array([[self.v_gear_lim[0]], [0]])
        )
        T.append(np.array([[self.v_gear_lim[1]], [-self.v_gear_lim[0]]]))
        T.append(np.array([[self.v_gear_lim[2]], [-self.v_gear_lim[1]]]))
        T.append(np.array([[self.alpha], [-self.v_gear_lim[2]]]))
        T.append(np.array([[self.v_gear_lim[3]], [-self.alpha]]))
        T.append(np.array([[self.v_gear_lim[4]], [-self.v_gear_lim[3]]]))
        T.append(
            np.array([[self.v_max], [-self.v_gear_lim[4]]])
            if bound_velocity
            else np.array([[0], [-self.v_gear_lim[4]]])
        )

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

        D = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        E = np.array([[self.p_max], [-self.p_min], [self.v_max], [-self.v_min]])
        F = np.array([[1], [-1]])
        G = np.array(
            [[self.u_max], [-self.u_min]]
        )  # velocity bounded -1 <= u <= 1 as it is normalized throttle

        return {
            "S": S,
            "R": R,
            "T": T,
            "A": A,
            "B": B,
            "c": c,
            "D": D,
            "E": E,
            "F": F,
            "G": G,
        }

    def get_gear_from_velocity(self, v: float):
        """Get the gear j from the velocity v as by the PWA model."""
        # check gear 2 to 5
        for i in range(len(self.b) - 2):
            if v >= self.v_gear_lim[i] and v < self.v_gear_lim[i + 1]:
                return i + 2

        # check gear 1
        if v < self.v_gear_lim[0]:
            if v < self.v_min:
                warnings.warn(
                    f"Velocity {v} is below min {self.v_min}, using first gear but result will be approximate."
                )
            return 1
        # check gear 6
        if v >= self.v_gear_lim[-1]:
            if v > self.v_max:
                warnings.warn(
                    f"Velocity {v} is above max {self.v_max}, using last gear but result will be approximate."
                )
            return 6
        raise RuntimeError(f"Didn't find any gear for the given speed {v}")

    def step_pwa(self, x: np.ndarray, u: np.ndarray, ts: float):
        """Steps the local PWA dynamics for a time step of ts seconds."""
        for j in range(len(self.system["S"])):
            if all(
                self.system["S"][j] @ x + self.system["R"][j] @ u
                <= self.system["T"][j]
                + np.array(
                    [[0], [1e-4]]
                )  # buffer is to have one of the regions as a strict inequality
            ):
                # perform a euler discretization of time-step t when stepping
                x_pwa = (
                    (np.eye(self.system["A"][j].shape[0]) + ts * self.system["A"][j])
                    @ x
                    + ts * self.system["B"][j] @ u
                    + ts * self.system["c"][j]
                )
                return x_pwa
        raise RuntimeError(f"Didnt find PWA region for x: {x} and u: {u}")

    def get_u_for_constant_vel(self, v: float):
        """Get the control input which will keep the velocity v constant, as by the PWA dynamics."""
        x = np.array([[0], [v]])  # first state does not matter for this pwa sys
        u = np.array([[0]])  # neither does control

        for j in range(len(self.system["S"])):
            if all(
                self.system["S"][j] @ x + self.system["R"][j] @ u
                <= self.system["T"][j]
                + np.array(
                    [[0], [1e-4]]
                )  # buffer is to have one of the as a strict inequality
            ):
                # This is VERY specific to this system, DO NOT reuse this code on other PWA systems.
                # we are setting the \dot{v} = 0, and solving for the input - using cont time model
                return (1 / self.system["B"][j][1, 0]) * (
                    -self.system["A"][j][1, 1] * v - self.system["c"][j][1, 0]
                )

        raise RuntimeError("Didn't find any PWA region for the given speed!")


if __name__ == "__main__":
    # check validity of PWA approximation
    np.random.seed(0)

    car_1 = Vehicle()
    car_2 = PwaGearVehicle()

    test_len = 100
    ts = 0.1
    x0 = np.array([[10], [15]])  # initial conditions
    u = (
        2 * np.random.random((1, test_len)) - 1
    )  # random control samples between -1 and 1
    x_1 = np.zeros((2, test_len))  # non-linear traj
    x_2 = np.zeros((2, test_len))  # pwa traj
    x_1[:, [0]] = x0
    x_2[:, [0]] = x0
    for t in range(test_len - 1):
        j = car_2.get_gear_from_velocity(
            x_2[1, [t]].item()
        )  # non-linear model uses same gear as PWA
        x_1[:, [t + 1]] = car_1.step(x_2[:, [t]], u[:, [t]].item(), j, ts)
        x_2[:, [t + 1]] = car_2.step_pwa(x_2[:, [t]], u[:, [t]], ts)

    _, axs = plt.subplots(2, 1, constrained_layout=True, sharex=True)
    axs[0].plot(x_1[0, :], "r")
    axs[0].plot(x_2[0, :], "b")
    axs[1].plot(x_1[1, :], "r")
    axs[1].plot(x_2[1, :], "b")
    plt.show()
