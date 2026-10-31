"""Constant-acceleration dynamics with selectable jerk-noise modelling.

The native grouped state ordering is

    x = [p_1, ..., p_n, v_1, ..., v_n, a_1, ..., a_n].

The deterministic constant-acceleration transition is

    p_next = p + dt * v + 0.5 * dt**2 * a
    v_next = v + dt * a
    a_next = a.

Two process-noise models are supported.

1. ``interval_jerk``

   A random jerk vector is held constant over one prediction interval:

       j_k ~ N(0, Sigma_j)

       w_k = G(dt) @ j_k

       G(dt) = [dt**3 / 6 * I;
                dt**2 / 2 * I;
                dt * I]

       Q(dt) = G(dt) @ Sigma_j @ G(dt).T.

   For coordinate i,

       Q_i = sigma_i**2 *
             [[dt**6 / 36, dt**5 / 12, dt**4 / 6],
              [dt**5 / 12, dt**4 / 4,  dt**3 / 2],
              [dt**4 / 6,  dt**3 / 2,  dt**2]].

   Here sigma_i is an interval-held jerk standard deviation.

2. ``white_jerk``

   The continuous-time SDE is

       dp = v dt
       dv = a dt
       da = sqrt(q) dW_t.

   Its exact discrete process covariance is

       Q_i = q_i *
             [[dt**5 / 20, dt**4 / 8, dt**3 / 6],
              [dt**4 / 8,  dt**3 / 3, dt**2 / 2],
              [dt**3 / 6,  dt**2 / 2, dt]].

   Here q_i is a continuous-time white-jerk noise intensity.

The deterministic transition matrix F is identical for both stochastic models.
Their noise parameters have different meanings and units and must be tuned
independently.
"""

import numpy as np

from .linear_base import LinearMotionModel


class ConstantAccelerationModel(LinearMotionModel):
    """Generic n-coordinate constant-acceleration model.

    Native state ordering
    ---------------------
    [p_1, ..., p_n, v_1, ..., v_n, a_1, ..., a_n]

    Examples
    --------
    dimensions=2:
        [px, py, vx, vy, ax, ay]

    dimensions=4:
        [cx, cy, width, height,
         dcx, dcy, dwidth, dheight,
         ddcx, ddcy, ddwidth, ddheight]

    Parameters
    ----------
    dimensions:
        Positive integer number of position-like coordinates.
        Total state dimension is ``3 * dimensions``.

    noise_model:
        Either ``"interval_jerk"`` or ``"white_jerk"``.

    jerk_std:
        Required for ``interval_jerk``.
        Nonnegative finite scalar or vector of shape ``(dimensions,)``.

        Entry i has units of coordinate_i / second**3.

    noise_intensity:
        Required for ``white_jerk``.
        Nonnegative finite scalar or vector of shape ``(dimensions,)``.

        Entry i is the continuous-time white-jerk noise intensity q_i.

    Notes
    -----
    Image-space coordinates such as width and height are not constrained to be
    positive by this model. Such constraints require an explicit policy or a
    different state parameterization.

    Noise arrays are copied on construction but remain writable for compatibility
    with existing callers. ``process_noise`` revalidates them before use.
    """

    name = "CA"

    def __init__(
        self,
        dimensions: int,
        *,
        noise_model: str = "interval_jerk",
        jerk_std=None,
        noise_intensity=None,
    ) -> None:
        if (
            isinstance(dimensions, (bool, np.bool_))
            or not isinstance(dimensions, (int, np.integer))
        ):
            raise TypeError("dimensions must be an integer.")

        if dimensions <= 0:
            raise ValueError("dimensions must be positive.")

        self.dimensions = int(dimensions)
        self.state_dim = 3 * self.dimensions
        self.noise_model = noise_model

        if noise_model == "interval_jerk":
            if jerk_std is None:
                raise ValueError(
                    "jerk_std is required for interval_jerk."
                )

            if noise_intensity is not None:
                raise ValueError(
                    "noise_intensity must not be supplied for interval_jerk."
                )

            self.jerk_std = self._validated_jerk_std(jerk_std)
            self.noise_intensity = None

        elif noise_model == "white_jerk":
            if noise_intensity is None:
                raise ValueError(
                    "noise_intensity is required for white_jerk."
                )

            if jerk_std is not None:
                raise ValueError(
                    "jerk_std must not be supplied for white_jerk."
                )

            self.noise_intensity = self._validated_noise_intensity(
                noise_intensity
            )
            self.jerk_std = None

        else:
            raise ValueError(
                "noise_model must be either 'interval_jerk' "
                "or 'white_jerk'."
            )

    def _validated_jerk_std(self, value) -> np.ndarray:
        """Validate interval-held jerk standard deviations."""
        return self._validated_vector_parameter(
            value,
            name="jerk_std",
            description="Jerk standard deviations",
        )

    def _validated_noise_intensity(self, value) -> np.ndarray:
        """Validate continuous-time white-jerk intensities."""
        return self._validated_vector_parameter(
            value,
            name="noise_intensity",
            description="Noise intensities",
        )

    def _validated_vector_parameter(
        self,
        value,
        *,
        name: str,
        description: str,
    ) -> np.ndarray:
        """Validate a nonnegative finite scalar/vector model parameter."""
        raw = np.asarray(value)

        # Reject bool, complex, strings, and arbitrary objects.
        if raw.dtype.kind not in "iuf":
            raise ValueError(
                f"{name} must contain real numerical values."
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
                f"{name} must be a scalar or an array "
                f"with shape ({self.dimensions},)."
            )

        if not np.all(np.isfinite(result)):
            raise ValueError(f"{description} must be finite.")

        if np.any(result < 0.0):
            raise ValueError(
                f"{description} must be non-negative."
            )

        return result.copy()

    def transition_matrix(self, dt: float) -> np.ndarray:
        """Return the grouped-state constant-acceleration transition matrix.

        Per coordinate,

            p_next = p + dt*v + 0.5*dt**2*a
            v_next = v + dt*a
            a_next = a.

        At dt = 0, F is the identity matrix.
        """
        dt = self.validate_dt(dt)
        n = self.dimensions

        try:
            with np.errstate(over="raise", invalid="raise"):
                dt2 = dt**2

                F = np.eye(3 * n, dtype=float)
                I = np.eye(n, dtype=float)

                position = slice(0, n)
                velocity = slice(n, 2 * n)
                acceleration = slice(2 * n, 3 * n)

                F[position, velocity] = dt * I
                F[position, acceleration] = 0.5 * dt2 * I
                F[velocity, acceleration] = dt * I

        except (OverflowError, FloatingPointError) as error:
            raise ValueError(
                "dt exceeds numerical transition range."
            ) from error

        if not np.all(np.isfinite(F)):
            raise ValueError(
                "dt exceeds numerical transition range."
            )

        return F

    def process_noise(self, dt: float) -> np.ndarray:
        """Return process-noise covariance for the selected CA noise model.

        ``interval_jerk``
            For coordinate i,

                sigma_i**2 *
                [[dt**6 / 36, dt**5 / 12, dt**4 / 6],
                 [dt**5 / 12, dt**4 / 4,  dt**3 / 2],
                 [dt**4 / 6,  dt**3 / 2,  dt**2]]

            corresponding to a jerk sample held constant over the prediction
            interval.

        ``white_jerk``
            For coordinate i,

                q_i *
                [[dt**5 / 20, dt**4 / 8, dt**3 / 6],
                 [dt**4 / 8,  dt**3 / 3, dt**2 / 2],
                 [dt**3 / 6,  dt**2 / 2, dt]]

            corresponding to exact discretization of continuous-time white jerk.

        Coordinate i occupies state indices

            position:     i
            velocity:     dimensions + i
            acceleration: 2*dimensions + i.
        """
        dt = self.validate_dt(dt)

        n = self.dimensions
        Q = np.zeros((3 * n, 3 * n), dtype=float)

        try:
            with np.errstate(over="raise", invalid="raise"):
                if self.noise_model == "interval_jerk":
                    sigma_values = self._validated_jerk_std(
                        self.jerk_std
                    )

                    dt2 = dt**2
                    dt3 = dt**3
                    dt4 = dt**4
                    dt5 = dt**5
                    dt6 = dt**6

                    for i, sigma in enumerate(sigma_values):
                        variance = sigma**2

                        p = i
                        v = n + i
                        a = 2 * n + i

                        Q[p, p] = (dt6 / 36.0) * variance

                        Q[p, v] = (dt5 / 12.0) * variance
                        Q[v, p] = Q[p, v]

                        Q[p, a] = (dt4 / 6.0) * variance
                        Q[a, p] = Q[p, a]

                        Q[v, v] = (dt4 / 4.0) * variance

                        Q[v, a] = (dt3 / 2.0) * variance
                        Q[a, v] = Q[v, a]

                        Q[a, a] = dt2 * variance

                elif self.noise_model == "white_jerk":
                    q_values = self._validated_noise_intensity(
                        self.noise_intensity
                    )

                    dt2 = dt**2
                    dt3 = dt**3
                    dt4 = dt**4
                    dt5 = dt**5

                    for i, q in enumerate(q_values):
                        p = i
                        v = n + i
                        a = 2 * n + i

                        Q[p, p] = (dt5 / 20.0) * q

                        Q[p, v] = (dt4 / 8.0) * q
                        Q[v, p] = Q[p, v]

                        Q[p, a] = (dt3 / 6.0) * q
                        Q[a, p] = Q[p, a]

                        Q[v, v] = (dt3 / 3.0) * q

                        Q[v, a] = (dt2 / 2.0) * q
                        Q[a, v] = Q[v, a]

                        Q[a, a] = dt * q

                else:
                    # Defensive guard if the public attribute is modified
                    # after construction.
                    raise ValueError(
                        f"Unknown noise model: {self.noise_model!r}"
                    )

        except (OverflowError, FloatingPointError) as error:
            raise ValueError(
                "Process-noise parameters and dt exceed numerical "
                "covariance range."
            ) from error

        if not np.all(np.isfinite(Q)):
            raise ValueError(
                "Process-noise covariance must be finite."
            )

        # Q is analytically symmetric; this suppresses only numerical asymmetry.
        return 0.5 * (Q + Q.T)


# FUTURE OPERATOR-BASED EXTENSION
# -------------------------------
# This CA model provides two explicit stochastic analytical baselines:
#
#   1. interval-held random jerk;
#   2. continuous-time white-jerk diffusion.
#
# They share the same deterministic CA transition F but imply different
# transition covariances Q(dt). Their noise parameters must be tuned
# independently because jerk_std and noise_intensity have different physical
# meanings and units.
#
# CA also represents acceleration explicitly, so comparisons with CV should
# distinguish the effect of state representation from the effect of stochastic
# transition modelling. These analytical models provide interpretable controls
# against which a learned conditional transition can be evaluated.
