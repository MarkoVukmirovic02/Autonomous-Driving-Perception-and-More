from abc import abstractmethod

import numpy as np

from ..measurement_base import MeasurementModel


class LinearMeasurementModel(MeasurementModel):
    """
    Base class for linear measurement models:

        z = Hx + v
    """

    @abstractmethod
    def measurement_matrix(self) -> np.ndarray:
        """Return the constant measurement matrix H."""
        raise NotImplementedError

    def predict_measurement(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        """
        Predict the ideal linear measurement:

            z_pred = Hx
        """
        state = self.validate_state(x)
        H = self.jacobian(state)

        return H @ state

    def jacobian(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        """
        For a linear measurement model:

            Jh(x) = H
        """
        self.validate_state(x)

        H = np.asarray(
            self.measurement_matrix(),
            dtype=float,
        )

        expected_shape = (
            self.measurement_dim,
            self.state_dim,
        )

        if H.shape != expected_shape:
            raise ValueError(
                f"{self.name} measurement matrix must have shape "
                f"{expected_shape}, but received {H.shape}."
            )

        if not np.all(np.isfinite(H)):
            raise ValueError(
                f"{self.name} measurement matrix contains "
                "NaN or infinite values."
            )

        return H.copy()