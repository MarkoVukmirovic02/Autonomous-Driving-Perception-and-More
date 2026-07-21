import numpy as np


class KalmanFilter:
    def __init__(
        self,
        x0: np.ndarray,
        P0: np.ndarray,
        motion_model,
        measurement_model,
    ):
        self.x = np.asarray(x0, dtype=float).reshape(-1)
        self.P = np.asarray(P0, dtype=float)

        self.motion_model = motion_model
        self.measurement_model = measurement_model

        self.I = np.eye(self.x.size, dtype=float)

        self._validate_dimensions()

        self.innovation = None
        self.innovation_covariance = None
        self.kalman_gain = None

    def _validate_dimensions(self) -> None:
        state_dim = self.x.size

        if self.P.shape != (state_dim, state_dim):
            raise ValueError(
                f"P0 must have shape {(state_dim, state_dim)}, "
                f"got {self.P.shape}."
            )

        if self.motion_model.state_dim != state_dim:
            raise ValueError(
                "Motion-model state dimension does not match x0."
            )

        H = self.measurement_model.measurement_matrix()
        R = self.measurement_model.measurement_noise()

        if H.shape[1] != state_dim:
            raise ValueError(
                "Measurement matrix column count must match state dimension."
            )

        measurement_dim = H.shape[0]

        if R.shape != (measurement_dim, measurement_dim):
            raise ValueError(
                "R must match the measurement dimension."
            )

    def predict(self, dt: float) -> np.ndarray:
        F = self.motion_model.state_transition_matrix(dt)
        Q = self.motion_model.process_noise(dt)

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

        # Remove small floating-point asymmetry.
        self.P = 0.5 * (self.P + self.P.T)

        return self.x.copy()

    def update(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float).reshape(-1)

        H = self.measurement_model.measurement_matrix()
        R = self.measurement_model.measurement_noise()

        if z.size != H.shape[0]:
            raise ValueError(
                f"Expected measurement dimension {H.shape[0]}, "
                f"got {z.size}."
            )

        y = z - H @ self.x

        # Later useful for angle measurements.
        y = self.measurement_model.normalize_residual(y)

        S = H @ self.P @ H.T + R

        # Equivalent to P H^T S^-1, but avoids explicit inversion.
        K = np.linalg.solve(S, H @ self.P).T

        self.x = self.x + K @ y

        I_KH = self.I - K @ H

        # Joseph form.
        self.P = (
            I_KH @ self.P @ I_KH.T
            + K @ R @ K.T
        )

        self.P = 0.5 * (self.P + self.P.T)

        self.innovation = y
        self.innovation_covariance = S
        self.kalman_gain = K

        return self.x.copy()