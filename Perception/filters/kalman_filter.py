import numpy as np

from Perception.motion_models.linear.linear_base import LinearMotionModel
from Perception.measurement_models.linear.linear_base import (
    LinearMeasurementModel,
)


class KalmanFilter:
    """
    Generic linear Kalman filter.

    The filter requires:

        - a linear motion model:

              x_k = F x_{k-1} + w_k

        - a linear measurement model:

              z_k = H x_k + v_k

    where:

        w_k ~ N(0, Q)
        v_k ~ N(0, R)
    """

    def __init__(
        self,
        x0: np.ndarray,
        P0: np.ndarray,
        motion_model: LinearMotionModel,
        measurement_model: LinearMeasurementModel,
    ) -> None:
        self.x = np.asarray(
            x0,
            dtype=float,
        ).reshape(-1).copy()

        self.P = np.asarray(
            P0,
            dtype=float,
        ).copy()

        if not isinstance(
            motion_model,
            LinearMotionModel,
        ):
            raise TypeError(
                "KalmanFilter requires a LinearMotionModel."
            )

        if not isinstance(
            measurement_model,
            LinearMeasurementModel,
        ):
            raise TypeError(
                "KalmanFilter requires a LinearMeasurementModel."
            )



        self.motion_model = motion_model
        self.measurement_model = measurement_model

        self.state_dim = self.x.size
        self.I = np.eye(
            self.state_dim,
            dtype=float,
        )

        self.innovation: np.ndarray | None = None
        self.innovation_covariance: np.ndarray | None = None
        self.kalman_gain: np.ndarray | None = None
        self.predicted_measurement: np.ndarray | None = None

        self._validate_initial_state()
        self._validate_models()

    def _validate_initial_state(self) -> None:
        """
        Validate x0 and P0.
        """
        if self.x.size == 0:
            raise ValueError(
                "x0 must contain at least one state variable."
            )

        if not np.all(np.isfinite(self.x)):
            raise ValueError(
                "x0 contains NaN or infinite values."
            )

        expected_shape = (
            self.state_dim,
            self.state_dim,
        )

        if self.P.shape != expected_shape:
            raise ValueError(
                f"P0 must have shape {expected_shape}, "
                f"but received {self.P.shape}."
            )

        if not np.all(np.isfinite(self.P)):
            raise ValueError(
                "P0 contains NaN or infinite values."
            )

        if not np.allclose(
            self.P,
            self.P.T,
            atol=1e-10,
        ):
            raise ValueError(
                "P0 must be symmetric."
            )

        eigenvalues = np.linalg.eigvalsh(self.P)

        if np.any(eigenvalues < -1e-10):
            raise ValueError(
                "P0 must be positive semidefinite."
            )

        # Remove insignificant numerical asymmetry.
        self.P = 0.5 * (
            self.P + self.P.T
        )

    def _validate_models(self) -> None:
        """
        Validate compatibility between the state, motion model,
        and measurement model.
        """
        if self.motion_model.state_dim != self.state_dim:
            raise ValueError(
                "Motion-model state dimension does not match x0: "
                f"{self.motion_model.state_dim} != {self.state_dim}."
            )

        if self.measurement_model.state_dim != self.state_dim:
            raise ValueError(
                "Measurement-model state dimension does not match x0: "
                f"{self.measurement_model.state_dim} != "
                f"{self.state_dim}."
            )

        H = np.asarray(
            self.measurement_model.measurement_matrix(),
            dtype=float,
        )

        R = np.asarray(
            self.measurement_model.measurement_noise(),
            dtype=float,
        )

        expected_H_shape = (
            self.measurement_model.measurement_dim,
            self.state_dim,
        )

        if H.shape != expected_H_shape:
            raise ValueError(
                f"H must have shape {expected_H_shape}, "
                f"but received {H.shape}."
            )

        expected_R_shape = (
            self.measurement_model.measurement_dim,
            self.measurement_model.measurement_dim,
        )

        if R.shape != expected_R_shape:
            raise ValueError(
                f"R must have shape {expected_R_shape}, "
                f"but received {R.shape}."
            )

        if not np.all(np.isfinite(H)):
            raise ValueError(
                "H contains NaN or infinite values."
            )

        if not np.all(np.isfinite(R)):
            raise ValueError(
                "R contains NaN or infinite values."
            )

        if not np.allclose(
            R,
            R.T,
            atol=1e-10,
        ):
            raise ValueError(
                "R must be symmetric."
            )

        eigenvalues = np.linalg.eigvalsh(R)

        if np.any(eigenvalues < -1e-10):
            raise ValueError(
                "R must be positive semidefinite."
            )

    def predict(
        self,
        dt: float,
    ) -> np.ndarray:
        """
        Perform the linear Kalman prediction step.
        """
        F = np.asarray(
            self.motion_model.transition_matrix(dt),
            dtype=float,
        )

        Q = np.asarray(
            self.motion_model.process_noise(dt),
            dtype=float,
        )

        expected_shape = (
            self.state_dim,
            self.state_dim,
        )

        if F.shape != expected_shape:
            raise ValueError(
                f"F must have shape {expected_shape}, "
                f"but received {F.shape}."
            )

        if Q.shape != expected_shape:
            raise ValueError(
                f"Q must have shape {expected_shape}, "
                f"but received {Q.shape}."
            )

        if not np.all(np.isfinite(F)):
            raise ValueError(
                "F contains NaN or infinite values."
            )

        if not np.all(np.isfinite(Q)):
            raise ValueError(
                "Q contains NaN or infinite values."
            )

        if not np.allclose(
            Q,
            Q.T,
            atol=1e-10,
        ):
            raise ValueError(
                "Q must be symmetric."
            )

        q_eigenvalues = np.linalg.eigvalsh(Q)

        if np.any(q_eigenvalues < -1e-10):
            raise ValueError(
                "Q must be positive semidefinite."
            )

        self.x = F @ self.x

        self.P = (
            F @ self.P @ F.T
            + Q
        )

        self.P = 0.5 * (
            self.P + self.P.T
        )

        if not np.all(np.isfinite(self.x)):
            raise FloatingPointError(
                "Predicted state contains NaN or infinite values."
            )

        if not np.all(np.isfinite(self.P)):
            raise FloatingPointError(
                "Predicted covariance contains NaN or infinite values."
            )

        return self.x.copy()

    def update(
        self,
        z: np.ndarray,
    ) -> np.ndarray:
        """
        Perform the linear Kalman correction step.

        The supplied z must already be in the measurement model's
        measurement space.

        For BoundingBoxMeasurementModel:

            z = [cx, cy, width, height]

        rather than:

            [x1, y1, x2, y2]
        """
        measurement = (
            self.measurement_model.validate_measurement(z)
        )

        H = np.asarray(
            self.measurement_model.measurement_matrix(),
            dtype=float,
        )

        R = np.asarray(
            self.measurement_model.measurement_noise(),
            dtype=float,
        )

        z_pred = (
            self.measurement_model.predict_measurement(
                self.x
            )
        )

        y = self.measurement_model.residual(
            measurement,
            z_pred,
        )

        S = H @ self.P @ H.T + R
        S = 0.5 * (S + S.T)

        if not np.all(np.isfinite(S)):
            raise FloatingPointError(
                "Innovation covariance contains NaN or infinite values."
        )

        # K = P H^T S^-1
        #
        # This solve avoids explicitly calculating inv(S).
        try:
            K = np.linalg.solve(
                S,
                H @ self.P,
            ).T
        except np.linalg.LinAlgError as error:
            raise np.linalg.LinAlgError(
                "Innovation covariance S is singular or "
                "numerically unstable."
            ) from error

        self.x = self.x + K @ y

        I_KH = self.I - K @ H

        # Joseph covariance update:
        #
        # P = (I-KH)P(I-KH)^T + KRK^T
        #
        # This is more numerically stable than:
        #
        # P = (I-KH)P
        self.P = (
            I_KH @ self.P @ I_KH.T
            + K @ R @ K.T
        )

        self.P = 0.5 * (
            self.P + self.P.T
        )


        if not np.all(np.isfinite(self.x)):
            raise FloatingPointError(
                "Updated state contains NaN or infinite values."
            )

        if not np.all(np.isfinite(self.P)):
            raise FloatingPointError(
                "Updated covariance contains NaN or infinite values."
            )

        self.predicted_measurement = z_pred.copy()
        self.innovation = y.copy()
        self.innovation_covariance = S.copy()
        self.kalman_gain = K.copy()

        return self.x.copy()

    def step(
        self,
        z: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """
        Perform prediction followed by measurement correction.
        """
        self.predict(dt)

        return self.update(z)