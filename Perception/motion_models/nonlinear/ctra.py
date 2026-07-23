import numpy as np


from .nonlinear_base import NonlinearMotionModel



class CTRAModel(NonlinearMotionModel):
    """
    Constant Turn Rate and Acceleration motion model.

    This implementation currently does not provide an analytical
    transition Jacobian. It is therefore directly suitable for UKF use,
    but not yet for EKF use.

    State ordering:

        x = [px, py, speed, acceleration, yaw, yaw_rate]

    where:

        px           : x position
        py           : y position
        speed        : longitudinal speed
        acceleration : longitudinal acceleration
        yaw          : heading angle in radians
        yaw_rate     : heading change rate in radians per second

    Model assumptions:

        - longitudinal acceleration is approximately constant;
        - yaw rate is approximately constant.

    Deviations from these assumptions are represented by process noise:

        - longitudinal jerk;
        - yaw acceleration.
    """

    name = "CTRA"
    state_dim = 6

    def __init__(
        self,
        jerk_std: float,
        yaw_acceleration_std: float,
        yaw_rate_epsilon: float = 1e-4,
    ) -> None:
        """
        Initialize the CTRA motion model.

        Parameters
        ----------
        jerk_std:
            Standard deviation of unmodeled longitudinal jerk.

            Jerk is the derivative of longitudinal acceleration:

                jerk = d(acceleration) / dt

        yaw_acceleration_std:
            Standard deviation of unmodeled yaw acceleration.

            Yaw acceleration is the derivative of yaw rate:

                yaw_acceleration = d(yaw_rate) / dt

        yaw_rate_epsilon:
            Positive numerical threshold used when yaw rate is close to zero.

            The exact CTRA position equations contain divisions by yaw_rate
            and yaw_rate**2. For sufficiently small absolute yaw rate, a
            Taylor-series approximation will be used instead.
        """
        if not np.isfinite(jerk_std):
            raise ValueError("jerk_std must be finite.")

        if jerk_std < 0.0:
            raise ValueError("jerk_std must be non-negative.")

        if not np.isfinite(yaw_acceleration_std):
            raise ValueError(
                "yaw_acceleration_std must be finite."
            )

        if yaw_acceleration_std < 0.0:
            raise ValueError(
                "yaw_acceleration_std must be non-negative."
            )

        if not np.isfinite(yaw_rate_epsilon):
            raise ValueError(
                "yaw_rate_epsilon must be finite."
            )

        if yaw_rate_epsilon <= 0.0:
            raise ValueError(
                "yaw_rate_epsilon must be positive."
            )

        self.jerk_std = float(jerk_std)
        self.yaw_acceleration_std = float(
            yaw_acceleration_std
        )
        self.yaw_rate_epsilon = float(
            yaw_rate_epsilon
        )

    def _predict_state(
        self,
        x: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """
        Predict the CTRA state after dt seconds.

        For nonzero yaw rate, the vehicle follows a curved trajectory
        while its speed changes linearly because acceleration is constant.

        For yaw rate close to zero, a Taylor-series approximation is used
        to avoid divisions by yaw_rate and yaw_rate**2.
        """

        px, py, speed, acceleration, yaw, yaw_rate = x

        yaw_new = yaw + yaw_rate * dt
        speed_new = speed + acceleration * dt

        if abs(yaw_rate) > self.yaw_rate_epsilon:
            sin_yaw = np.sin(yaw)
            cos_yaw = np.cos(yaw)
            sin_yaw_new = np.sin(yaw_new)
            cos_yaw_new = np.cos(yaw_new)

            inverse_yaw_rate = 1.0 / yaw_rate
            inverse_yaw_rate_squared = inverse_yaw_rate**2

            px_new = (
                px
                + speed
                * inverse_yaw_rate
                * (sin_yaw_new - sin_yaw)
                + acceleration
                * dt
                * inverse_yaw_rate
                * sin_yaw_new
                + acceleration
                * inverse_yaw_rate_squared
                * (cos_yaw_new - cos_yaw)
            )

            py_new = (
                py
                + speed
                * inverse_yaw_rate
                * (cos_yaw - cos_yaw_new)
                - acceleration
                * dt
                * inverse_yaw_rate
                * cos_yaw_new
                + acceleration
                * inverse_yaw_rate_squared
                * (sin_yaw_new - sin_yaw)
            )

        else:
            dt2 = dt**2
            dt3 = dt**3
            dt4 = dt**4

            sin_yaw = np.sin(yaw)
            cos_yaw = np.cos(yaw)

            # Taylor expansion around yaw_rate = 0.
            #
            # These expressions reduce to straight-line constant
            # acceleration when yaw_rate is exactly zero.
            px_increment = (
                speed * cos_yaw * dt
                + 0.5 * acceleration * cos_yaw * dt2
                - 0.5 * speed * sin_yaw * yaw_rate * dt2
                - (1.0 / 3.0)
                * acceleration
                * sin_yaw
                * yaw_rate
                * dt3
                - (1.0 / 6.0)
                * speed
                * cos_yaw
                * yaw_rate**2
                * dt3
                - (1.0 / 8.0)
                * acceleration
                * cos_yaw
                * yaw_rate**2
                * dt4
            )

            py_increment = (
                speed * sin_yaw * dt
                + 0.5 * acceleration * sin_yaw * dt2
                + 0.5 * speed * cos_yaw * yaw_rate * dt2
                + (1.0 / 3.0)
                * acceleration
                * cos_yaw
                * yaw_rate
                * dt3
                - (1.0 / 6.0)
                * speed
                * sin_yaw
                * yaw_rate**2
                * dt3
                - (1.0 / 8.0)
                * acceleration
                * sin_yaw
                * yaw_rate**2
                * dt4
            )

            px_new = px + px_increment
            py_new = py + py_increment

        return np.array(
            [
                px_new,
                py_new,
                speed_new,
                acceleration,
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
        Construct the CTRA process-noise covariance Q.

        State ordering:
            [px, py, speed, acceleration, yaw, yaw_rate]

        Random inputs:
            longitudinal jerk
            yaw acceleration

        Longitudinal jerk is projected onto the global x and y axes
        using the current heading.
        """

        yaw = x[4]

        dt2 = dt**2
        dt3 = dt**3

        G = np.array(
            [
                [(dt3 / 6.0) * np.cos(yaw), 0.0],
                [(dt3 / 6.0) * np.sin(yaw), 0.0],
                [0.5 * dt2,                 0.0],
                [dt,                        0.0],
                [0.0,               0.5 * dt2],
                [0.0,                        dt],
            ],
            dtype=float,
        )

        noise_covariance = np.diag(
            [
                self.jerk_std**2,
                self.yaw_acceleration_std**2,
            ]
        )

        Q = G @ noise_covariance @ G.T

        # Remove tiny numerical asymmetry.
        return 0.5 * (Q + Q.T)
    
