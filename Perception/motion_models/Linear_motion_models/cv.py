from .linear_base import LinearMotionModel
import numpy as np

class ConstantVelocityModel(LinearMotionModel):
    """
    Generic n-dimensional Constant Velocity model.

    State ordering:

        [p_1, p_2, ..., p_n, v_1, v_2, ..., v_n]

    Examples:

        [x, y, vx, vy]

        [cx, cy, w, h, vx, vy, vw, vh]
    """

    name='CV'

    def __init__(
        self,
        dimensions: int,
        acceleration_std: float | np.ndarray,
    ):
        if dimensions <= 0:
            raise ValueError("dimensions must be positive.")

        self.dimensions = dimensions
        self.state_dim = 2 * dimensions

        acceleration_std = np.asarray(
            acceleration_std,
            dtype=float,
        )

        if acceleration_std.ndim == 0:
            acceleration_std = np.full(
                dimensions,
                float(acceleration_std),
            )

        if acceleration_std.shape != (dimensions,):
            raise ValueError(
                "acceleration_std must be a scalar or an array "
                f"with shape ({dimensions},)."
            )

        if np.any(acceleration_std < 0.0):
            raise ValueError(
                "Acceleration standard deviations must be non-negative."
            )

        self.acceleration_std = acceleration_std

    def transition_matrix(self, dt: float) -> np.ndarray:
        """
        Construct the Constant Velocity transition matrix F.
        """
        if dt < 0.0:
            raise ValueError("dt must be non-negative.")

        n = self.dimensions

        F = np.eye(2 * n, dtype=float)

        F[:n, n:] = dt * np.eye(n)

        return F

    def process_noise(self, dt: float) -> np.ndarray:
        """
        Construct the CV process-noise covariance Q.

        Each position/rate pair uses

            sigma_i^2 *
            [[dt^4 / 4, dt^3 / 2],
             [dt^3 / 2, dt^2]]
        """
        if dt < 0.0:
            raise ValueError("dt must be non-negative.")

        n = self.dimensions
        Q = np.zeros((2 * n, 2 * n), dtype=float)

        dt2 = dt**2
        dt3 = dt**3
        dt4 = dt**4

        for i, sigma in enumerate(self.acceleration_std):
            variance = sigma**2

            p = i
            v = n + i

            Q[p, p] = 0.25 * dt4 * variance
            Q[p, v] = 0.5 * dt3 * variance
            Q[v, p] = 0.5 * dt3 * variance
            Q[v, v] = dt2 * variance

        return Q
