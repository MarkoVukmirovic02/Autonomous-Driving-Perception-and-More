"""Planar CTRA dynamics with selectable process-noise modelling.

The native state is

    x = [px, py, v, a, yaw, yaw_rate].

Nominal continuous-time dynamics are

    dpx/dt       = v * cos(yaw)
    dpy/dt       = v * sin(yaw)
    dv/dt        = a
    da/dt        = 0
    dyaw/dt      = yaw_rate
    dyaw_rate/dt = 0.

Acceleration is longitudinal. Even with constant longitudinal acceleration,
turning produces Cartesian acceleration components through the changing
heading. Signed speed is permitted; negative v means motion opposite the yaw
parameter. No separate body-orientation, sideslip, or nonnegative-speed
constraint is imposed.

Two process-noise models are supported.

1. ``interval_jerk``

   Longitudinal jerk and angular acceleration are treated as independent random
   variables held constant over one prediction interval. A frozen-heading
   disturbance mapping constructs

       Q = G @ diag(sigma_j**2, sigma_alpha**2) @ G.T.

2. ``continuous_diffusion``

   Longitudinal acceleration and yaw rate are driven by continuous-time
   diffusion:

       da        = sqrt(q_j)     dW_j
       dyaw_rate = sqrt(q_alpha) dW_alpha.

   The nonlinear drift is locally linearized at the supplied state and the
   resulting frozen linear SDE is discretized with the Van Loan method.

The nominal transition and analytical Jacobian use the same stable integrated
arc representation. The two noise models have different parameter meanings and
units and must be tuned independently.
"""

import numpy as np
from scipy.linalg import expm

from .nonlinear_base import NonlinearMotionModel


def _scalar(value, name, *, positive=False):
    """Validate and return a finite nonnegative real scalar."""
    raw = np.asarray(value)

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


class CTRAModel(NonlinearMotionModel):
    """Planar constant-turn-rate and longitudinal-acceleration model.

    State
    -----
    [px, py, v, a, yaw, yaw_rate]

    px, py:
        Position in a fixed planar frame.

    v:
        Signed longitudinal speed.

    a:
        Signed longitudinal acceleration.

    yaw:
        Heading parameter in radians, wrapped to [-pi, pi).

    yaw_rate:
        Heading rate in rad/s.

    Process-noise choices
    ---------------------
    ``interval_jerk``:
        Uses interval-held longitudinal jerk and angular acceleration.

        Required parameters:
            jerk_std
            yaw_acceleration_std

    ``continuous_diffusion``:
        Uses continuous-time diffusion in longitudinal acceleration and yaw rate.

        Required parameters:
            jerk_noise_intensity
            yaw_rate_noise_intensity

    Notes
    -----
    ``yaw_rate_epsilon`` is a numerical threshold used by the stable arc-moment
    evaluation. It is not a physical minimum turn rate.

    Noise parameters remain writable for compatibility with existing callers
    and are revalidated whenever process-noise covariance is constructed.
    """

    name = "CTRA"
    state_dim = 6
    angle_indices = (4,)

    def __init__(
        self,
        *,
        noise_model: str = "interval_jerk",
        jerk_std=None,
        yaw_acceleration_std=None,
        jerk_noise_intensity=None,
        yaw_rate_noise_intensity=None,
        yaw_rate_epsilon: float = 1e-4,
    ) -> None:
        self.noise_model = noise_model

        self.yaw_rate_epsilon = _scalar(
            yaw_rate_epsilon,
            "yaw_rate_epsilon",
            positive=True,
        )

        if noise_model == "interval_jerk":
            if jerk_std is None:
                raise ValueError(
                    "jerk_std is required for interval_jerk."
                )

            if yaw_acceleration_std is None:
                raise ValueError(
                    "yaw_acceleration_std is required for interval_jerk."
                )

            if (
                jerk_noise_intensity is not None
                or yaw_rate_noise_intensity is not None
            ):
                raise ValueError(
                    "Continuous-time noise intensities must not be supplied "
                    "for interval_jerk."
                )

            self.jerk_std = _scalar(
                jerk_std,
                "jerk_std",
            )
            self.yaw_acceleration_std = _scalar(
                yaw_acceleration_std,
                "yaw_acceleration_std",
            )

            self.jerk_noise_intensity = None
            self.yaw_rate_noise_intensity = None

        elif noise_model == "continuous_diffusion":
            if jerk_noise_intensity is None:
                raise ValueError(
                    "jerk_noise_intensity is required for "
                    "continuous_diffusion."
                )

            if yaw_rate_noise_intensity is None:
                raise ValueError(
                    "yaw_rate_noise_intensity is required for "
                    "continuous_diffusion."
                )

            if jerk_std is not None or yaw_acceleration_std is not None:
                raise ValueError(
                    "Interval-held noise standard deviations must not be "
                    "supplied for continuous_diffusion."
                )

            self.jerk_noise_intensity = _scalar(
                jerk_noise_intensity,
                "jerk_noise_intensity",
            )
            self.yaw_rate_noise_intensity = _scalar(
                yaw_rate_noise_intensity,
                "yaw_rate_noise_intensity",
            )

            self.jerk_std = None
            self.yaw_acceleration_std = None

        else:
            raise ValueError(
                "noise_model must be either 'interval_jerk' "
                "or 'continuous_diffusion'."
            )

    def _arc_moments(self, rate, dt):
        """Return J_n(theta) for n=0,1,2 with theta = rate*dt.

        J_n(theta) is

            integral_0^1 s**n exp(i*theta*s) ds.

        Complex arithmetic compactly represents planar rotation while all
        public state, covariance, and Jacobian outputs remain real.

        Near zero turn angle, the convergent power series

            J_n = sum_m (i*theta)**m / (m! * (n+m+1))

        avoids inverse-rate cancellation. Away from zero, integration by parts
        provides a compact stable recurrence.
        """
        epsilon = _scalar(
            self.yaw_rate_epsilon,
            "yaw_rate_epsilon",
            positive=True,
        )

        theta = rate * dt

        if not np.isfinite(theta):
            raise ValueError("Turn angle exceeds numerical range.")

        z = 1j * theta

        if abs(theta) <= 0.5 or (
            abs(rate) <= epsilon and abs(theta) <= 1.0
        ):
            moments = np.zeros(3, dtype=complex)
            term = 1.0 + 0.0j

            for m in range(24):
                moments += term / np.array(
                    [m + 1.0, m + 2.0, m + 3.0]
                )
                term *= z / (m + 1.0)

        else:
            exponential = np.exp(z)
            moments = np.empty(3, dtype=complex)

            moments[0] = np.expm1(z) / z

            for n in (1, 2):
                moments[n] = (
                    exponential - n * moments[n - 1]
                ) / z

        return moments

    @staticmethod
    def _planar(value):
        """Convert a complex planar quantity to [x, y]."""
        return np.array(
            [value.real, value.imag],
            dtype=float,
        )

    def _predict_state(self, x: np.ndarray, dt: float) -> np.ndarray:
        """Advance one state using the stable integrated CTRA arc.

        Position displacement is

            exp(i*yaw) *
            (v*dt*J_0 + a*dt**2*J_1).

        At zero yaw rate,

            J_0 = 1,
            J_1 = 1/2,

        recovering straight-line constant-acceleration motion.

        At a = 0, position evolution reduces to CTRV.
        """
        px, py, v, a, yaw, rate = x

        J0, J1, _ = self._arc_moments(rate, dt)

        rotation = np.exp(1j * yaw)
        displacement = rotation * (
            v * dt * J0
            + a * dt**2 * J1
        )

        return np.array(
            [
                px + displacement.real,
                py + displacement.imag,
                v + a * dt,
                a,
                self.normalize_angle(yaw + rate * dt),
                rate,
            ],
            dtype=float,
        )

    def _process_noise(self, x: np.ndarray, dt: float) -> np.ndarray:
        """Return process-noise covariance for the selected CTRA noise model.

        ``interval_jerk``
            Uses interval-held longitudinal jerk j and angular acceleration
            alpha with frozen heading:

                delta_position =
                    dt**3/6 * j *
                    [cos(yaw), sin(yaw)]

                delta_v =
                    dt**2/2 * j

                delta_a =
                    dt * j

                delta_yaw =
                    dt**2/2 * alpha

                delta_yaw_rate =
                    dt * alpha.

            This is a discrete frozen-heading disturbance approximation.

        ``continuous_diffusion``
            Uses the continuous-time SDE

                dpx       = v*cos(yaw) dt
                dpy       = v*sin(yaw) dt
                dv        = a dt
                da        = sqrt(q_j) dW_j
                dyaw      = yaw_rate dt
                dyaw_rate = sqrt(q_alpha) dW_alpha.

            The nonlinear drift is locally linearized at x. The resulting
            frozen linear SDE is then discretized with the Van Loan matrix
            exponential. The result is exact for that local linear system,
            not for the full nonlinear SDE.
        """
        state = self.validate_state(x)
        dt = self.validate_dt(dt)

        if self.noise_model == "interval_jerk":
            return self._process_noise_interval(state, dt)

        if self.noise_model == "continuous_diffusion":
            return self._process_noise_continuous(state, dt)

        raise ValueError(
            f"Unknown noise model: {self.noise_model!r}"
        )

    def _process_noise_interval(
        self,
        x: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """Construct frozen-heading interval-held jerk covariance."""
        sigma_j = _scalar(
            self.jerk_std,
            "jerk_std",
        )
        sigma_alpha = _scalar(
            self.yaw_acceleration_std,
            "yaw_acceleration_std",
        )

        yaw = x[4]

        try:
            with np.errstate(over="raise", invalid="raise"):
                dt2 = dt**2
                dt3 = dt**3

                G = np.array(
                    [
                        [dt3 / 6.0 * np.cos(yaw), 0.0],
                        [dt3 / 6.0 * np.sin(yaw), 0.0],
                        [0.5 * dt2, 0.0],
                        [dt, 0.0],
                        [0.0, 0.5 * dt2],
                        [0.0, dt],
                    ],
                    dtype=float,
                )

                input_covariance = np.diag(
                    [
                        sigma_j**2,
                        sigma_alpha**2,
                    ]
                )

                Q = G @ input_covariance @ G.T

        except (OverflowError, FloatingPointError) as error:
            raise ValueError(
                "Noise parameters and dt exceed numerical covariance range."
            ) from error

        if not np.all(np.isfinite(Q)):
            raise ValueError(
                "Process-noise covariance must be finite."
            )

        return 0.5 * (Q + Q.T)

    def _process_noise_continuous(
        self,
        x: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """Discretize the locally linearized continuous CTRA diffusion.

        The frozen drift Jacobian is

            A =
            [[0, 0, cos(yaw), 0, -v*sin(yaw), 0],
             [0, 0, sin(yaw), 0,  v*cos(yaw), 0],
             [0, 0,        0, 1,             0, 0],
             [0, 0,        0, 0,             0, 0],
             [0, 0,        0, 0,             0, 1],
             [0, 0,        0, 0,             0, 0]]

        with continuous diffusion covariance

            S = diag(0, 0, 0, q_j, 0, q_alpha).

        Van Loan discretization returns the exact discrete covariance for this
        frozen local linear system.
        """
        q_j = _scalar(
            self.jerk_noise_intensity,
            "jerk_noise_intensity",
        )
        q_alpha = _scalar(
            self.yaw_rate_noise_intensity,
            "yaw_rate_noise_intensity",
        )

        _, _, v, _, yaw, _ = x

        c = np.cos(yaw)
        s = np.sin(yaw)

        A = np.array(
            [
                [0.0, 0.0, c,   0.0, -v * s, 0.0],
                [0.0, 0.0, s,   0.0,  v * c, 0.0],
                [0.0, 0.0, 0.0, 1.0,     0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0,     0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0,     0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0,     0.0, 0.0],
            ],
            dtype=float,
        )

        S = np.diag(
            [
                0.0,
                0.0,
                0.0,
                q_j,
                0.0,
                q_alpha,
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
                "Continuous CTRA diffusion and dt exceed numerical "
                "discretization range."
            ) from error

        if not np.all(np.isfinite(Q)):
            raise ValueError(
                "Process-noise covariance must be finite."
            )

        return 0.5 * (Q + Q.T)

    def jacobian(self, x: np.ndarray, dt: float) -> np.ndarray:
        """Return the local Jacobian of the nominal discrete CTRA transition.

        With complex displacement D and R = exp(i*yaw),

            dD/dv =
                R * dt * J_0

            dD/da =
                R * dt**2 * J_1

            dD/dyaw =
                i * D

            dD/dyaw_rate =
                i * R *
                (v*dt**2*J_1 + a*dt**3*J_2).

        Real and imaginary parts form the two position rows. The expression
        has a continuous zero-rate limit. Yaw differentiation is local and
        excludes the wrapping branch cut.
        """
        state = self.validate_state(x)
        dt = self.validate_dt(dt)

        _, _, v, a, yaw, rate = state

        J0, J1, J2 = self._arc_moments(rate, dt)
        rotation = np.exp(1j * yaw)

        dv = rotation * dt * J0
        da = rotation * dt**2 * J1
        displacement = v * dv + a * da
        drate = (
            1j
            * rotation
            * (
                v * dt**2 * J1
                + a * dt**3 * J2
            )
        )

        F = np.eye(self.state_dim, dtype=float)

        F[:2, 2] = self._planar(dv)
        F[:2, 3] = self._planar(da)
        F[:2, 4] = self._planar(
            1j * displacement
        )
        F[:2, 5] = self._planar(drate)

        F[2, 3] = dt
        F[4, 5] = dt

        return self.validate_jacobian(F)


# FUTURE OPERATOR-BASED EXTENSION
# -------------------------------
# CTRA is a richer nonlinear analytical state model, not automatically a better
# estimator. Acceleration observability, state conversion uncertainty, and the
# adequacy of the Markov state must be evaluated separately.
#
# The interval-held and continuous-diffusion process models provide two explicit
# stochastic baselines for the same nominal CTRA dynamics. A learned conditional
# transition may replace these analytical transition assumptions while Bayesian
# prediction remains the operation that propagates the belief.
#
# CTRA participation in IMM also requires explicit state-conversion adapters and
# covariance mappings between CTRA and the other motion-model state spaces.
