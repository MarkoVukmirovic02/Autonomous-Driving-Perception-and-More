from abc import ABC, abstractmethod

import numpy as np


class MeasurementModel(ABC):
    """Common interface for all sensor measurement models."""

    name: str
    state_dim: int
    measurement_dim: int

    def validate_state(self, x: np.ndarray) -> np.ndarray:
        state = (
            np.asarray(
                x,
                dtype=float,
            )
            .reshape(-1)
            .copy()
        )

        if state.shape != (self.state_dim,):
            raise ValueError(
                f"{self.name} expected state shape "
                f"({self.state_dim},), got {state.shape}."
            )

        if not np.all(np.isfinite(state)):
            raise ValueError(
                f"{self.name} received a state containing NaN or infinity."
            )

        return state

    def validate_measurement(self, z: np.ndarray) -> np.ndarray:
        measurement = (
            np.asarray(
                z,
                dtype=float,
            )
            .reshape(-1)
            .copy()
        )

        if measurement.shape != (self.measurement_dim,):
            raise ValueError(
                f"{self.name} expected measurement shape "
                f"({self.measurement_dim},), got {measurement.shape}."
            )

        if not np.all(np.isfinite(measurement)):
            raise ValueError(
                f"{self.name} received a measurement containing NaN or infinity."
            )

        return measurement

    @abstractmethod
    def predict_measurement(self, x: np.ndarray) -> np.ndarray:
        """Return the ideal predicted measurement h(x)."""
        raise NotImplementedError

    @abstractmethod
    def measurement_noise(self) -> np.ndarray:
        """Return the measurement-noise covariance R."""
        raise NotImplementedError

    def residual(
        self,
        z: np.ndarray,
        z_pred: np.ndarray,
    ) -> np.ndarray:
        """Compute z - h(x), with sensor-specific normalization if needed."""
        measurement = self.validate_measurement(z)
        prediction = self.validate_measurement(z_pred)

        return measurement - prediction