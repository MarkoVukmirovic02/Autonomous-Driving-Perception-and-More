import numpy as np

from .nonlinear_base import NonlinearMotionModel


class CTRVModel(NonlinearMotionModel):
    """
    Coordinated Turn Rate and Velocity motion model.

    State ordering:

        x = [px, py, v, yaw, yaw_rate]

    where:

        px       : x position
        py       : y position
        v        : speed magnitude
        yaw      : heading angle in radians
        yaw_rate : heading change rate in radians per second

    Model assumptions:

        - speed is approximately constant;
        - yaw rate is approximately constant.

    Deviations from these assumptions are represented through process noise:

        - longitudinal acceleration;
        - yaw acceleration.
    """

    name = "CTRV"
    state_dim = 5

    def __init__(
        self,
        acceleration_std: float,
        yaw_acceleration_std: float,
        yaw_rate_epsilon: float = 1e-4,
    ) -> None:
        if not np.isfinite(acceleration_std):
            raise ValueError("acceleration_std must be finite.")

        if acceleration_std < 0.0:
            raise ValueError("acceleration_std must be non-negative.")

        if not np.isfinite(yaw_acceleration_std):
            raise ValueError("yaw_acceleration_std must be finite.")

        if yaw_acceleration_std < 0.0:
            raise ValueError(
                "yaw_acceleration_std must be non-negative."
            )

        if not np.isfinite(yaw_rate_epsilon):
            raise ValueError("yaw_rate_epsilon must be finite.")

        if yaw_rate_epsilon <= 0.0:
            raise ValueError("yaw_rate_epsilon must be positive.")

        self.acceleration_std = float(acceleration_std)
        self.yaw_acceleration_std = float(yaw_acceleration_std)
        self.yaw_rate_epsilon = float(yaw_rate_epsilon)

    def _predict_state(
        self,
        x: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """
        Predict the CTRV state after dt seconds.

        For nonzero yaw rate, the object follows a circular arc.

        For yaw rate close to zero, the circular equations are replaced
        with their straight-line limit to avoid division by a small number.
        """

        px, py, speed, yaw, yaw_rate = x

        yaw_new = yaw + yaw_rate * dt

        if abs(yaw_rate) > self.yaw_rate_epsilon:
            px_new = px + (speed / yaw_rate) * (
                np.sin(yaw_new) - np.sin(yaw)
            )

            py_new = py + (speed / yaw_rate) * (
                np.cos(yaw) - np.cos(yaw_new)
            )

        else:
            dt2 = dt**2
            dt3 = dt**3

            cos_yaw = np.cos(yaw)
            sin_yaw = np.sin(yaw)

            px_new = px + speed * (
                cos_yaw * dt
                - 0.5 * sin_yaw * yaw_rate * dt2
                - (1.0 / 6.0) * cos_yaw * yaw_rate**2 * dt3
            )

            py_new = py + speed * (
                sin_yaw * dt
                + 0.5 * cos_yaw * yaw_rate * dt2
                - (1.0 / 6.0) * sin_yaw * yaw_rate**2 * dt3
            )

        return np.array(
            [
                px_new,
                py_new,
                speed,
                self.normalize_angle(yaw_new),
                yaw_rate,
            ],
            dtype=float,
        )
    
    def _process_noise(
        self,
        x: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        
        """
        Construct the CTRV process-noise covariance Q.

        State ordering:
            [px, py, speed, yaw, yaw_rate]

        Random inputs:
            longitudinal acceleration
            yaw acceleration

        The acceleration is projected onto the global x and y axes
        according to the current heading.
        """



        yaw = x[3]

        half_dt_squared = 0.5 * dt**2

        # Maps the two random inputs
        #
        #     [longitudinal_acceleration, yaw_acceleration]
        #
        # into the five-dimensional CTRV state.
        G = np.array(
            [
                [half_dt_squared * np.cos(yaw), 0.0],
                [half_dt_squared * np.sin(yaw), 0.0],
                [dt,                              0.0],
                [0.0,                 half_dt_squared],
                [0.0,                              dt],
            ],
            dtype=float,
        )

        noise_covariance = np.diag(
            [
                self.acceleration_std**2,
                self.yaw_acceleration_std**2,
            ]
        )

        Q = G @ noise_covariance @ G.T

        # Protect against tiny floating-point asymmetry.
        return 0.5 * (Q + Q.T)
    

    def jacobian(
        self,
        x: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """
        Return the CTRV transition Jacobian df/dx.

        State ordering:
            [px, py, speed, yaw, yaw_rate]
        """
        state = self.validate_state(x)
        dt = self.validate_dt(dt)

        _, _, speed, yaw, yaw_rate = state

        if abs(yaw_rate) > self.yaw_rate_epsilon:
            yaw_new = yaw + yaw_rate * dt

            sin_yaw = np.sin(yaw)
            cos_yaw = np.cos(yaw)
            sin_yaw_new = np.sin(yaw_new)
            cos_yaw_new = np.cos(yaw_new)

            inverse_yaw_rate = 1.0 / yaw_rate
            inverse_yaw_rate_squared = inverse_yaw_rate**2

            dpx_dspeed = inverse_yaw_rate * (
                sin_yaw_new - sin_yaw
            )

            dpx_dyaw = speed * inverse_yaw_rate * (
                cos_yaw_new - cos_yaw
            )

            dpx_dyaw_rate = (
                speed * dt * inverse_yaw_rate * cos_yaw_new
                - speed
                * inverse_yaw_rate_squared
                * (sin_yaw_new - sin_yaw)
            )

            dpy_dspeed = inverse_yaw_rate * (
                cos_yaw - cos_yaw_new
            )

            dpy_dyaw = speed * inverse_yaw_rate * (
                sin_yaw_new - sin_yaw
            )

            dpy_dyaw_rate = (
                speed * dt * inverse_yaw_rate * sin_yaw_new
                - speed
                * inverse_yaw_rate_squared
                * (cos_yaw - cos_yaw_new)
            )

            F = np.array(
                [
                    [
                        1.0,
                        0.0,
                        dpx_dspeed,
                        dpx_dyaw,
                        dpx_dyaw_rate,
                    ],
                    [
                        0.0,
                        1.0,
                        dpy_dspeed,
                        dpy_dyaw,
                        dpy_dyaw_rate,
                    ],
                    [0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, dt],
                    [0.0, 0.0, 0.0, 0.0, 1.0],
                ],
                dtype=float,
            )

            return self.validate_jacobian(F)

        # Straight-line limit for yaw_rate near zero.
        # Exact yaw-rate -> 0 limit of the CTRV Jacobian.
        # Taylor-series Jacobian for yaw_rate near zero.
        dt2 = dt**2
        dt3 = dt**3

        sin_yaw = np.sin(yaw)
        cos_yaw = np.cos(yaw)

        dpx_dspeed = (
            cos_yaw * dt
            - 0.5 * sin_yaw * yaw_rate * dt2
            - (1.0 / 6.0) * cos_yaw * yaw_rate**2 * dt3
        )

        dpx_dyaw = speed * (
            -sin_yaw * dt
            - 0.5 * cos_yaw * yaw_rate * dt2
            + (1.0 / 6.0) * sin_yaw * yaw_rate**2 * dt3
        )

        dpx_dyaw_rate = speed * (
            -0.5 * sin_yaw * dt2
            - (1.0 / 3.0) * cos_yaw * yaw_rate * dt3
        )

        dpy_dspeed = (
            sin_yaw * dt
            + 0.5 * cos_yaw * yaw_rate * dt2
            - (1.0 / 6.0) * sin_yaw * yaw_rate**2 * dt3
        )

        dpy_dyaw = speed * (
            cos_yaw * dt
            - 0.5 * sin_yaw * yaw_rate * dt2
            - (1.0 / 6.0) * cos_yaw * yaw_rate**2 * dt3
        )

        dpy_dyaw_rate = speed * (
            0.5 * cos_yaw * dt2
            - (1.0 / 3.0) * sin_yaw * yaw_rate * dt3
        )

        F = np.array(
            [
                [
                    1.0,
                    0.0,
                    dpx_dspeed,
                    dpx_dyaw,
                    dpx_dyaw_rate,
                ],
                [
                    0.0,
                    1.0,
                    dpy_dspeed,
                    dpy_dyaw,
                    dpy_dyaw_rate,
                ],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, dt],
                [0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

        return self.validate_jacobian(F)