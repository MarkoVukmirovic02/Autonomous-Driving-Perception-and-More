"""Planar CTRV dynamics with selectable process-noise modelling.

The native state is

    x = [px, py, v, yaw, yaw_rate].

Nominal continuous-time dynamics are

    dpx/dt      = v * cos(yaw)
    dpy/dt      = v * sin(yaw)
    dv/dt       = 0
    dyaw/dt     = yaw_rate
    dyaw_rate/dt = 0

The implementation permits signed v so Gaussian estimates and sigma points are
not clipped at zero. For v >= 0, yaw is the velocity direction; for v < 0,
motion is opposite yaw. There is no separate body-orientation or slip state.

Two process-noise models are supported:

1. ``interval_acceleration``

   Longitudinal acceleration and angular acceleration are treated as independent
   random variables held constant over one prediction interval. A frozen-heading
   mapping is used to construct

       Q = G @ diag(sigma_a**2, sigma_alpha**2) @ G.T.

2. ``continuous_diffusion``

   Signed speed and yaw rate are driven by continuous-time white-noise
   diffusions,

       dv       = sqrt(q_v)     dW_v
       dyaw_rate = sqrt(q_omega) dW_omega.

   The nonlinear drift is locally linearized at the supplied state and the
   resulting frozen linear SDE is discretized with the Van Loan method.

The deterministic CTRV transition is identical for both process-noise choices.
The two parameterizations have different meanings and units and must be tuned
independently.
"""

import numpy as np
from scipy.linalg import expm

from .nonlinear_base import NonlinearMotionModel


def _scalar(value, name, *, positive=False):
    """Validate and return a finite nonnegative real scalar."""
    raw = np.asarray(value)

    # bool, complex, string and object values are intentionally rejected.
    if raw.ndim != 0 or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a real numerical scalar.")

    value = float(raw)

    invalid = (
        not np.isfinite(value)
        or value < 0.0
        or (positive and value == 0.0)
    )
    if invalid:
        bound = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be finite and {bound}.")

    return value


class CTRVModel(NonlinearMotionModel):
    """Planar constant-turn-rate and constant-speed-magnitude motion model.

    State
    -----
    [px, py, v, yaw, yaw_rate]

    px, py:
        Position in a fixed planar frame.

    v:
        Signed speed.

    yaw:
        Heading parameter in radians. It is wrapped to [-pi, pi).

    yaw_rate:
        Heading rate in rad/s. Positive values turn from +x toward +y under
        the usual mathematical convention.

    Process-noise choices
    ---------------------
    ``interval_acceleration``:
        Uses interval-held longitudinal acceleration and angular acceleration.

        Required parameters:
            acceleration_std
            yaw_acceleration_std

    ``continuous_diffusion``:
        Uses continuous-time diffusion in signed speed and yaw rate.

        Required parameters:
            speed_noise_intensity
            yaw_rate_noise_intensity

    Notes
    -----
    ``yaw_rate_epsilon`` is a numerical threshold used only by the stable arc
    evaluation near zero turn angle. It is not a physical minimum turn rate.

    Noise parameters remain writable for compatibility with existing callers.
    They are revalidated whenever process-noise covariance is constructed.
    """

    name = "CTRV"
    state_dim = 5
    angle_indices = (3,)

    def __init__(
        self,
        *,
        noise_model: str = "interval_acceleration",
        acceleration_std=None,
        yaw_acceleration_std=None,
        speed_noise_intensity=None,
        yaw_rate_noise_intensity=None,
        yaw_rate_epsilon: float = 1e-4,
    ) -> None:
        self.noise_model = noise_model

        self.yaw_rate_epsilon = _scalar(
            yaw_rate_epsilon,
            "yaw_rate_epsilon",
            positive=True,
        )

        if noise_model == "interval_acceleration":
            if acceleration_std is None:
                raise ValueError(
                    "acceleration_std is required for interval_acceleration."
                )

            if yaw_acceleration_std is None:
                raise ValueError(
                    "yaw_acceleration_std is required for "
                    "interval_acceleration."
                )

            if (
                speed_noise_intensity is not None
                or yaw_rate_noise_intensity is not None
            ):
                raise ValueError(
                    "Continuous-time noise intensities must not be supplied "
                    "for interval_acceleration."
                )

            self.acceleration_std = _scalar(
                acceleration_std,
                "acceleration_std",
            )
            self.yaw_acceleration_std = _scalar(
                yaw_acceleration_std,
                "yaw_acceleration_std",
            )

            self.speed_noise_intensity = None
            self.yaw_rate_noise_intensity = None

        elif noise_model == "continuous_diffusion":
            if speed_noise_intensity is None:
                raise ValueError(
                    "speed_noise_intensity is required for "
                    "continuous_diffusion."
                )

            if yaw_rate_noise_intensity is None:
                raise ValueError(
                    "yaw_rate_noise_intensity is required for "
                    "continuous_diffusion."
                )

            if (
                acceleration_std is not None
                or yaw_acceleration_std is not None
            ):
                raise ValueError(
                    "Interval-held acceleration standard deviations must not "
                    "be supplied for continuous_diffusion."
                )

            self.speed_noise_intensity = _scalar(
                speed_noise_intensity,
                "speed_noise_intensity",
            )
            self.yaw_rate_noise_intensity = _scalar(
                yaw_rate_noise_intensity,
                "yaw_rate_noise_intensity",
            )

            self.acceleration_std = None
            self.yaw_acceleration_std = None

        else:
            raise ValueError(
                "noise_model must be either 'interval_acceleration' "
                "or 'continuous_diffusion'."
            )

    def _arc_terms(self, yaw_rate, dt):
        """Return u, sinc(u), and d sinc(u)/du for u = yaw_rate*dt/2.

        Here sinc(u) = sin(u)/u, not NumPy's normalized sinc convention.
        Series expansions avoid cancellation around u = 0.
        """
        epsilon = _scalar(
            self.yaw_rate_epsilon,
            "yaw_rate_epsilon",
            positive=True,
        )

        u = 0.5 * yaw_rate * dt

        if not np.isfinite(u):
            raise ValueError("Turn angle exceeds numerical range.")

        if abs(u) <= 1e-3 or (
            abs(yaw_rate) <= epsilon and abs(u) <= 1e-2
        ):
            u2 = u * u
            s = 1.0 + u2 * (
                -1.0 / 6.0
                + u2 * (
                    1.0 / 120.0
                    + u2 * (-1.0 / 5040.0 + u2 / 362880.0)
                )
            )
            ds = u * (
                -1.0 / 3.0
                + u2 * (
                    1.0 / 30.0
                    + u2 * (-1.0 / 840.0 + u2 / 45360.0)
                )
            )
        else:
            s = np.sin(u) / u
            ds = (np.cos(u) - s) / u

        return u, s, ds

    def _predict_state(self, x: np.ndarray, dt: float) -> np.ndarray:
        """Advance one state using a stable circular-arc representation.

        displacement =
            v * dt * sinc(u) *
            [cos(yaw + u), sin(yaw + u)]

        with u = yaw_rate * dt / 2.

        This is equivalent to the standard CTRV circular-arc equations but
        remains numerically well behaved as yaw_rate approaches zero.
        """
        px, py, v, yaw, rate = x

        u, s, _ = self._arc_terms(rate, dt)
        mid = yaw + u
        distance = v * dt * s

        return np.array(
            [
                px + distance * np.cos(mid),
                py + distance * np.sin(mid),
                v,
                self.normalize_angle(yaw + 2.0 * u),
                rate,
            ],
            dtype=float,
        )

    def _process_noise(self, x: np.ndarray, dt: float) -> np.ndarray:
        """Return process-noise covariance for the selected CTRV noise model.

        ``interval_acceleration``
            Uses interval-held longitudinal acceleration a and angular
            acceleration alpha:

                delta_p =
                    0.5 * dt**2 * a *
                    [cos(yaw), sin(yaw)]

                delta_v =
                    dt * a

                delta_yaw =
                    0.5 * dt**2 * alpha

                delta_yaw_rate =
                    dt * alpha

            This is a frozen-heading discrete disturbance model.

        ``continuous_diffusion``
            Uses the continuous-time SDE

                dpx       = v*cos(yaw) dt
                dpy       = v*sin(yaw) dt
                dv        = sqrt(q_v) dW_v
                dyaw      = yaw_rate dt
                dyaw_rate = sqrt(q_omega) dW_omega

            The nonlinear drift is locally linearized at x. The resulting
            frozen linear SDE is then discretized using the Van Loan matrix
            exponential. The discretization is exact for that frozen local
            linear system, not for the full nonlinear SDE.
        """
        state = self.validate_state(x)
        dt = self.validate_dt(dt)

        if self.noise_model == "interval_acceleration":
            return self._process_noise_interval(state, dt)

        if self.noise_model == "continuous_diffusion":
            return self._process_noise_continuous(state, dt)

        # Defensive guard if the public attribute is changed after construction.
        raise ValueError(f"Unknown noise model: {self.noise_model!r}")

    def _process_noise_interval(
        self,
        x: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """Construct frozen-heading interval-held acceleration covariance."""
        sigma_a = _scalar(
            self.acceleration_std,
            "acceleration_std",
        )
        sigma_alpha = _scalar(
            self.yaw_acceleration_std,
            "yaw_acceleration_std",
        )

        yaw = x[3]

        try:
            with np.errstate(over="raise", invalid="raise"):
                half_dt2 = 0.5 * dt**2

                G = np.array(
                    [
                        [half_dt2 * np.cos(yaw), 0.0],
                        [half_dt2 * np.sin(yaw), 0.0],
                        [dt, 0.0],
                        [0.0, half_dt2],
                        [0.0, dt],
                    ],
                    dtype=float,
                )

                input_covariance = np.diag(
                    [
                        sigma_a**2,
                        sigma_alpha**2,
                    ]
                )

                Q = G @ input_covariance @ G.T

        except (OverflowError, FloatingPointError) as error:
            raise ValueError(
                "Noise parameters and dt exceed numerical covariance range."
            ) from error

        if not np.all(np.isfinite(Q)):
            raise ValueError("Process-noise covariance must be finite.")

        # Remove only roundoff-level asymmetry.
        return 0.5 * (Q + Q.T)

    def _process_noise_continuous(
        self,
        x: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """Discretize the locally linearized continuous CTRV diffusion.

        q_v drives signed speed and q_omega drives yaw rate.

        The frozen continuous-time Jacobian is

            A =
            [[0, 0, cos(yaw), -v*sin(yaw), 0],
             [0, 0, sin(yaw),  v*cos(yaw), 0],
             [0, 0,        0,             0, 0],
             [0, 0,        0,             0, 1],
             [0, 0,        0,             0, 0]]

        with continuous diffusion covariance

            S = diag(0, 0, q_v, 0, q_omega).

        Van Loan discretization returns the exact discrete covariance for this
        frozen local linear system.
        """
        q_v = _scalar(
            self.speed_noise_intensity,
            "speed_noise_intensity",
        )
        q_omega = _scalar(
            self.yaw_rate_noise_intensity,
            "yaw_rate_noise_intensity",
        )

        _, _, v, yaw, _ = x

        c = np.cos(yaw)
        s = np.sin(yaw)

        A = np.array(
            [
                [0.0, 0.0, c, -v * s, 0.0],
                [0.0, 0.0, s,  v * c, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=float,
        )

        S = np.diag(
            [
                0.0,
                0.0,
                q_v,
                0.0,
                q_omega,
            ]
        )

        try:
            with np.errstate(over="raise", invalid="raise"):
                n = self.state_dim

                van_loan = np.zeros(
                    (2 * n, 2 * n),
                    dtype=float,
                )

                # Van Loan construction:
                #
                # exp(
                #     [[ A, S],
                #      [ 0,-A.T]] * dt
                # )
                #
                # = [[Phi, M],
                #    [   0, Phi^-T]]
                #
                # and Q = M @ Phi.T.

                van_loan[:n, :n] = A
                van_loan[:n, n:] = S
                van_loan[n:, n:] = -A.T

                E = expm(van_loan * dt)

                Phi = E[:n, :n]
                M = E[:n, n:]

                Q = M @ Phi.T

        except (OverflowError, FloatingPointError) as error:
            raise ValueError(
                "Continuous CTRV diffusion and dt exceed numerical "
                "discretization range."
            ) from error

        if not np.all(np.isfinite(Q)):
            raise ValueError("Process-noise covariance must be finite.")

        # Numerical matrix exponentiation can introduce tiny asymmetry.
        return 0.5 * (Q + Q.T)

    def jacobian(self, x: np.ndarray, dt: float) -> np.ndarray:
        """Return the local derivative of the nominal discrete CTRV transition.

        With

            u   = yaw_rate * dt / 2
            mid = yaw + u

        the displacement derivatives are

            d displacement / dv
                = dt * sinc(u) * direction(mid)

            d displacement / dyaw
                = v * dt * sinc(u) * normal(mid)

            d displacement / dyaw_rate
                = v * dt**2 / 2 *
                  [sinc'(u) * direction(mid)
                   + sinc(u) * normal(mid)].

        Angular differentiation uses a local unwrapped chart; the wrapped yaw
        has a branch cut.
        """
        state = self.validate_state(x)
        dt = self.validate_dt(dt)

        _, _, v, yaw, rate = state

        u, s, ds = self._arc_terms(rate, dt)
        mid = yaw + u

        direction = np.array(
            [np.cos(mid), np.sin(mid)],
            dtype=float,
        )
        normal = np.array(
            [-np.sin(mid), np.cos(mid)],
            dtype=float,
        )

        F = np.eye(self.state_dim, dtype=float)

        F[:2, 2] = dt * s * direction
        F[:2, 3] = v * dt * s * normal
        F[:2, 4] = (
            v
            * dt
            * (0.5 * dt)
            * (ds * direction + s * normal)
        )
        F[3, 4] = dt

        return self.validate_jacobian(F)


# FUTURE OPERATOR-BASED EXTENSION
# -------------------------------
# CTRV supplies a nonlinear turning benchmark with explicit heading and
# yaw-rate states. The analytical process models above provide interpretable
# controls against which a learned conditional transition can be evaluated.
#
# Changes in state representation should be distinguished from changes in the
# learned transition law. A learned operator on a richer belief representation
# is not equivalent to multiplying physical coordinates by a state-transition
# Jacobian. Signed-speed conventions and angular charts must also remain
# consistent when CTRV participates in IMM model conversion and mixing.
