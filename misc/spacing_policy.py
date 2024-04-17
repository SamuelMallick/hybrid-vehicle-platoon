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
        self.d = np.array([[-d0, 0]]).T

    def spacing(self, x: np.ndarray):
        return self.d


class ConstantTimePolicy(SpacingPolicy):
    """A spacing policy that forces a constant time seperation between vehicles with velocity dependent spacing."""

    def __init__(self, d0: float, t0: float) -> None:
        super().__init__()
        # spacing is A@x + b
        self.A = np.array([[0, -t0], [0, 0]])
        self.b = np.array([[-d0], [0]])

    def spacing(self, x: np.ndarray):
        if len(x.shape) == 1:
            x = x.reshape(x.shape[0], 1)
        return self.A @ x + self.b
