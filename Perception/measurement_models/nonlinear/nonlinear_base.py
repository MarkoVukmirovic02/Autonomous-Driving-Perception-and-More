from abc import abstractmethod

import numpy as np

from ..measurement_base import MeasurementModel


class NonlinearMeasurementModel(MeasurementModel):
    """
    Base class for nonlinear measurement models.

    Nonlinear measurement equation:

        z = h(x) + v

    where:

        x : hidden state vector
        z : sensor measurement
        h : nonlinear measurement function
        v ~ N(0, R)
        R : measurement-noise covariance
    """

    def predict_measurement(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        """
        Validate the state and evaluate the nonlinear measurement
        function h(x).
        """
        state = self.validate_state(x)

        prediction = np.asarray(
            self._predict_measurement(state),
            dtype=float,
        ).reshape(-1)

        if prediction.shape != (self.measurement_dim,):
            raise ValueError(
                f"{self.name} predicted measurement must have shape "
                f"({self.measurement_dim},), but received "
                f"{prediction.shape}."
            )

        if not np.all(np.isfinite(prediction)):
            raise ValueError(
                f"{self.name} predicted measurement contains "
                "NaN or infinite values."
            )

        return prediction.copy()

    @abstractmethod
    def _predict_measurement(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        """
        Evaluate h(x) using an already validated state vector.
        """
        raise NotImplementedError

    @abstractmethod
    def measurement_noise(self) -> np.ndarray:
        """
        Return the measurement-noise covariance matrix R.
        """
        raise NotImplementedError

    def jacobian(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        """
        Return the measurement Jacobian:

            J_h(x) = dh(x) / dx

        EKF-compatible subclasses must override this method.
        UKF-only subclasses do not need to.
        """
        self.validate_state(x)

        raise NotImplementedError(
            f"{self.name} does not provide a measurement Jacobian."
        )

    def validate_jacobian(
        self,
        jacobian: np.ndarray,
    ) -> np.ndarray:
        """
        Validate and return a measurement Jacobian.
        """
        J = np.asarray(
            jacobian,
            dtype=float,
        )

        expected_shape = (
            self.measurement_dim,
            self.state_dim,
        )

        if J.shape != expected_shape:
            raise ValueError(
                f"{self.name} Jacobian must have shape "
                f"{expected_shape}, but received {J.shape}."
            )

        if not np.all(np.isfinite(J)):
            raise ValueError(
                f"{self.name} Jacobian contains NaN or infinite values."
            )

        return J.copy()