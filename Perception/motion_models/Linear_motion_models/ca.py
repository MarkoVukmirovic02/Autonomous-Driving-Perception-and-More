import numpy as np
from .linear_base import LinearMotionModel


class ConstantAccelerationModel(LinearMotionModel):

    """
    Generic n-dimensional Constant Acceleration motion model.

    State ordering:

        [p_1, p_2, ..., p_n,
         v_1, v_2, ..., v_n,
         a_1, a_2, ..., a_n]

    Examples
    --------
    Two-dimensional Cartesian tracking:

        [x, y, vx, vy, ax, ay]

    Bounding-box tracking:

        [cx, cy, w, h,
         vx, vy, vw, vh,
         ax, ay, aw, ah]

    Model assumption
    ----------------
    Acceleration remains constant during one prediction interval.

    The unmodeled input is jerk, which describes how quickly
    acceleration changes.
    """

    name = "CA"

    def __init__(
        self,
        dimensions: int,
        jerk_std: float | np.ndarray,
    ):
        if dimensions <= 0:
            raise ValueError("dimensions must be positive.")

        self.dimensions = dimensions
        self.state_dim = 3 * dimensions

        jerk_std = np.asarray(
            jerk_std,
            dtype=float,
        )

        # Allow one common jerk standard deviation for every dimension.
        if jerk_std.ndim == 0:
            jerk_std = np.full(
                dimensions,
                float(jerk_std),
            )

        if jerk_std.shape != (dimensions,):
            raise ValueError(
                "jerk_std must be a scalar or an array "
                f"with shape ({dimensions},)."
            )

        if np.any(jerk_std < 0.0):
            raise ValueError(
                "Jerk standard deviations must be non-negative."
            )

        self.jerk_std = jerk_std

    def transition_matrix(self, dt: float) -> np.ndarray:
        """
        Construct the Constant Acceleration transition matrix F.

        For every dimension i:

            p_i' = p_i + v_i * dt + 0.5 * a_i * dt^2
            v_i' = v_i + a_i * dt
            a_i' = a_i
        """
        if not np.isfinite(dt):
            raise ValueError("dt must be finite.")

        if dt < 0.0:
            raise ValueError("dt must be non-negative.")

        n = self.dimensions
        dt2 = dt**2

        F = np.eye(3 * n, dtype=float)

        position_slice = slice(0, n)
        velocity_slice = slice(n, 2 * n)
        acceleration_slice = slice(2 * n, 3 * n)

        identity = np.eye(n, dtype=float)

        # Position depends on velocity.
        F[position_slice, velocity_slice] = dt * identity

        # Position also depends on acceleration.
        F[position_slice, acceleration_slice] = (
            0.5 * dt2 * identity
        )

        # Velocity depends on acceleration.
        F[velocity_slice, acceleration_slice] = dt * identity

        return F

    def process_noise(self, dt: float) -> np.ndarray:
        """
        Construct the CA process-noise covariance Q.

        CA assumes acceleration is constant, so unknown jerk is the
        process-noise input.

        For each dimension, jerk affects:

            delta_position     = (1/6) * jerk * dt^3
            delta_velocity     = (1/2) * jerk * dt^2
            delta_acceleration = jerk * dt

        Therefore, for each [position_i, velocity_i, acceleration_i]
        group:

            Q_i = sigma_jerk_i^2 *

                [[dt^6 / 36, dt^5 / 12, dt^4 / 6],
                 [dt^5 / 12, dt^4 / 4,  dt^3 / 2],
                 [dt^4 / 6,  dt^3 / 2,  dt^2    ]]

        Independent jerk noise is assumed across dimensions.
        """
        if not np.isfinite(dt):
            raise ValueError("dt must be finite.")

        if dt < 0.0:
            raise ValueError("dt must be non-negative.")

        n = self.dimensions
        Q = np.zeros((3 * n, 3 * n), dtype=float)

        dt2 = dt**2
        dt3 = dt**3
        dt4 = dt**4
        dt5 = dt**5
        dt6 = dt**6

        for i, sigma in enumerate(self.jerk_std):
            variance = sigma**2

            position_index = i
            velocity_index = n + i
            acceleration_index = 2 * n + i

            Q[position_index, position_index] = (
                dt6 / 36.0 * variance
            )
            Q[position_index, velocity_index] = (
                dt5 / 12.0 * variance
            )
            Q[position_index, acceleration_index] = (
                dt4 / 6.0 * variance
            )

            Q[velocity_index, position_index] = (
                dt5 / 12.0 * variance
            )
            Q[velocity_index, velocity_index] = (
                dt4 / 4.0 * variance
            )
            Q[velocity_index, acceleration_index] = (
                dt3 / 2.0 * variance
            )

            Q[acceleration_index, position_index] = (
                dt4 / 6.0 * variance
            )
            Q[acceleration_index, velocity_index] = (
                dt3 / 2.0 * variance
            )
            Q[acceleration_index, acceleration_index] = (
                dt2 * variance
            )

        return Q