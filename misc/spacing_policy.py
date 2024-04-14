import numpy as np


class SpacingPolicy:
    """Class that implements a general inter-vehicle spacing policy."""

    def spacing(self, x: np.ndarray):
        """Get desired spacing between vehicles"""
        raise NotImplementedError(
            "Spacing function only implemented in sub classes of SpacingPolicy"
        )


class ConstantSpacingPolicy(SpacingPolicy):
    """A constant spacing policy."""

    def __init__(self, d0: float) -> None:
        super().__init__()
        self.d0 = d0

    def spacing(self, x: np.ndarray):
        return np.array([[-self.d0], [0]])


class ConstantTimePolicy(SpacingPolicy):
    """A spacing policy that forces a constant time seperation between vehicles with velocity dependent spacing."""

    def __init__(self, d0: float, t0: float) -> None:
        super().__init__()
        self.d0 = d0
        self.t0 = t0

    def spacing(self, x: np.ndarray):
        return np.array([[-self.t0 * x[1, 0] - self.d0], [0]])
