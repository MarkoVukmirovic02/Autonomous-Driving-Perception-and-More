"""Constant-velocity dynamics with selectable process-noise modelling.

The deterministic CV transition is

    p_next = p + dt * v
    v_next = v

with grouped state ordering

    [p_1, ..., p_n, v_1, ..., v_n].

Two stochastic process-noise models are supported.

1. Interval-held random acceleration

    a_k ~ N(0, Sigma_a)

    w_k = G(dt) @ a_k

    G(dt) = [0.5 * dt**2 * I;
             dt * I]

which gives

    Q(dt) = G(dt) @ Sigma_a @ G(dt).T.

For coordinate i:

    Q_i = sigma_i**2 *
          [[dt**4 / 4, dt**3 / 2],
           [dt**3 / 2, dt**2]]

Here sigma_i is an acceleration standard deviation with units
coordinate / second**2.

2. Continuous-time white-acceleration SDE

    dp = v dt
    dv = sqrt(q) dW_t

whose exact discrete covariance is

    Q_i = q_i *
          [[dt**3 / 3, dt**2 / 2],
           [dt**2 / 2, dt]]

Here q_i is a continuous-time process-noise intensity and does not
have the same interpretation or units as sigma_i**2 above.

The deterministic transition matrix F is identical for both models.
"""

import numpy as np

from .linear_base import LinearMotionModel


class ConstantVelocityModel(LinearMotionModel):
    """Generic n-coordinate constant-velocity model.

    Native state ordering:

        [p_1, ..., p_n, v_1, ..., v_n].

    Examples
    --------
    dimensions=2:
        [px, py, vx, vy]

    dimensions=4:
        [cx, cy, width, height,
         dcx, dcy, dwidth, dheight]

    Parameters
    ----------
    dimensions:
        Positive number of position-like coordinates.
        Total state dimension is 2 * dimensions.

    noise_model:
        Either

            "interval_acceleration"

        or

            "white_acceleration".

    acceleration_std:
        Required for ``interval_acceleration``.
        Nonnegative scalar or vector of shape (dimensions,).

    noise_intensity:
        Required for ``white_acceleration``.
        Nonnegative scalar or vector of shape (dimensions,).
        Represents continuous-time white-noise intensity q.
    """

    name = "CV"

    def __init__(
        self,
        dimensions: int,
        *,
        noise_model: str = "interval_acceleration",
        acceleration_std=None,
        noise_intensity=None,
    ):
        # bool is technically an int subclass, but is not meaningful here.
        if (
            isinstance(dimensions, (bool, np.bool_))
            or not isinstance(dimensions, (int, np.integer))
        ):
            raise TypeError("dimensions must be an integer.")

        if dimensions <= 0:
            raise ValueError("dimensions must be positive.")

        self.dimensions = int(dimensions)
        self.state_dim = 2 * self.dimensions
        self.noise_model = noise_model

        if noise_model == "interval_acceleration":
            if acceleration_std is None:
                raise ValueError(
                    "acceleration_std is required for "
                    "interval_acceleration."
                )

            if noise_intensity is not None:
                raise ValueError(
                    "noise_intensity must not be supplied for "
                    "interval_acceleration."
                )

            self.acceleration_std = (
                self._validated_acceleration_std(acceleration_std)
            )
            self.noise_intensity = None

        elif noise_model == "white_acceleration":
            if noise_intensity is None:
                raise ValueError(
                    "noise_intensity is required for "
                    "white_acceleration."
                )

            if acceleration_std is not None:
                raise ValueError(
                    "acceleration_std must not be supplied for "
                    "white_acceleration."
                )

            self.noise_intensity = (
                self._validated_noise_intensity(noise_intensity)
            )
            self.acceleration_std = None

        else:
            raise ValueError(
                "noise_model must be either "
                "'interval_acceleration' or 'white_acceleration'."
            )

    def _validated_acceleration_std(self, value):
        """Validate interval-held acceleration standard deviations."""
        raw = np.asarray(value)

        if raw.dtype.kind not in "iuf":
            raise ValueError(
                "acceleration_std must contain real numerical values."
            )

        result = np.asarray(raw, dtype=float)

        if result.ndim == 0:
            result = np.full(
                self.dimensions,
                float(result),
                dtype=float,
            )

        if result.shape != (self.dimensions,):
            raise ValueError(
                "acceleration_std must be a scalar or an array "
                f"with shape ({self.dimensions},)."
            )

        if not np.all(np.isfinite(result)):
            raise ValueError(
                "Acceleration standard deviations must be finite."
            )

        if np.any(result < 0):
            raise ValueError(
                "Acceleration standard deviations must be non-negative."
            )

        return result.copy()

    def _validated_noise_intensity(self, value):
        """Validate continuous-time white-acceleration intensities."""
        raw = np.asarray(value)

        if raw.dtype.kind not in "iuf":
            raise ValueError(
                "noise_intensity must contain real numerical values."
            )

        result = np.asarray(raw, dtype=float)

        if result.ndim == 0:
            result = np.full(
                self.dimensions,
                float(result),
                dtype=float,
            )

        if result.shape != (self.dimensions,):
            raise ValueError(
                "noise_intensity must be a scalar or an array "
                f"with shape ({self.dimensions},)."
            )

        if not np.all(np.isfinite(result)):
            raise ValueError(
                "Noise intensities must be finite."
            )

        if np.any(result < 0):
            raise ValueError(
                "Noise intensities must be non-negative."
            )

        return result.copy()

    def transition_matrix(self, dt: float) -> np.ndarray:
        """Return F = [[I, dt I], [0, I]]."""
        dt = self.validate_dt(dt)

        n = self.dimensions

        F = np.eye(2 * n, dtype=float)
        F[:n, n:] = dt * np.eye(n, dtype=float)

        return F

    def process_noise(self, dt: float) -> np.ndarray:
        """Return process-noise covariance for the selected model.

        For ``interval_acceleration``:

            sigma_i**2 *
            [[dt**4 / 4, dt**3 / 2],
             [dt**3 / 2, dt**2]]

        For ``white_acceleration``:

            q_i *
            [[dt**3 / 3, dt**2 / 2],
             [dt**2 / 2, dt]]

        Coordinate i occupies state indices

            position: i
            rate:     dimensions + i.
        """
        dt = self.validate_dt(dt)

        n = self.dimensions
        Q = np.zeros((2 * n, 2 * n), dtype=float)

        try:
            with np.errstate(over="raise", invalid="raise"):

                if self.noise_model == "interval_acceleration":
                    sigma_values = (
                        self._validated_acceleration_std(
                            self.acceleration_std
                        )
                    )

                    dt2 = dt**2
                    dt3 = dt**3
                    dt4 = dt**4

                    for i, sigma in enumerate(sigma_values):
                        variance = sigma**2

                        p = i
                        v = n + i

                        Q[p, p] = 0.25 * dt4 * variance
                        Q[p, v] = 0.5 * dt3 * variance
                        Q[v, p] = 0.5 * dt3 * variance
                        Q[v, v] = dt2 * variance

                elif self.noise_model == "white_acceleration":
                    q_values = (
                        self._validated_noise_intensity(
                            self.noise_intensity
                        )
                    )

                    dt2 = dt**2
                    dt3 = dt**3

                    for i, q in enumerate(q_values):
                        p = i
                        v = n + i

                        Q[p, p] = (dt3 / 3.0) * q
                        Q[p, v] = (dt2 / 2.0) * q
                        Q[v, p] = (dt2 / 2.0) * q
                        Q[v, v] = dt * q

                else:
                    # Defensive check if the public attribute was modified.
                    raise ValueError(
                        f"Unknown noise model: {self.noise_model!r}"
                    )

        except (OverflowError, FloatingPointError) as error:
            raise ValueError(
                "Process-noise parameters exceed numerical "
                "covariance range."
            ) from error

        if not np.all(np.isfinite(Q)):
            raise ValueError(
                "Process-noise covariance must be finite."
            )

        return Q


# FUTURE OPERATOR-BASED EXTENSION
# -------------------------------
# This CV model provides two explicit stochastic analytical baselines:
#
#   1. interval-held random acceleration;
#   2. continuous-time white-acceleration diffusion.
#
# They share the same deterministic CV transition F but imply different
# stochastic transition laws Q(dt).
#
# Their respective noise parameters must be tuned independently because
# acceleration_std and noise_intensity have different physical meanings
# and units.
#
# Controlled trajectories can be used to determine whether either
# stochastic assumption adequately represents the target dynamics before
# comparing them with learned conditional-transition approaches.
#
# A learned operator on belief coefficients is distinct from the physical
# state transition matrix F. Learning may alter the conditional transition
# representation while Bayesian prediction remains the operation used to
# propagate the previous belief.