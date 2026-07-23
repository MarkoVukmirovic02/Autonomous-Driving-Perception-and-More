from abc import ABC, abstractmethod

import numpy as np


class NonlinearMotionModel(ABC):
    """
    Base class for nonlinear motion models.

    Nonlinear state-transition equation:

        x_k = f(x_{k-1}, dt) + w_k

    where:

        w_k ~ N(0, Q(x, dt))
    """

    name: str
    state_dim: int

    @staticmethod
    def normalize_angle(angle: float) -> float:
        """Normalize an angle to [-pi, pi)."""
        return (angle + np.pi) % (2.0 * np.pi) - np.pi

    def validate_state(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        """Return a validated one-dimensional state vector."""
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
                f"{self.name} state must have shape "
                f"({self.state_dim},), but received {state.shape}."
            )

        if not np.all(np.isfinite(state)):
            raise ValueError(
                f"{self.name} state contains NaN or infinite values."
            )

        return state

    @staticmethod
    def validate_dt(dt: float) -> float:
        """Validate and return the prediction interval."""
        dt = float(dt)

        if not np.isfinite(dt):
            raise ValueError("dt must be finite.")

        if dt < 0.0:
            raise ValueError("dt must be non-negative.")

        return dt

    def predict_state(
        self,
        x: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """
        Validate the inputs and evaluate the nonlinear transition:

            x_pred = f(x, dt)
        """
        state = self.validate_state(x)
        dt = self.validate_dt(dt)

        prediction = np.asarray(
            self._predict_state(state, dt),
            dtype=float,
        ).reshape(-1)

        if prediction.shape != (self.state_dim,):
            raise ValueError(
                f"{self.name} predicted state must have shape "
                f"({self.state_dim},), but received "
                f"{prediction.shape}."
            )

        if not np.all(np.isfinite(prediction)):
            raise ValueError(
                f"{self.name} predicted state contains "
                "NaN or infinite values."
            )

        return prediction.copy()

    @abstractmethod
    def _predict_state(
        self,
        x: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """
        Evaluate f(x, dt) using validated inputs.
        """
        raise NotImplementedError

    def process_noise(
        self,
        x: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """
        Validate the inputs and construct Q(x, dt).
        """
        state = self.validate_state(x)
        dt = self.validate_dt(dt)

        Q = np.asarray(
            self._process_noise(state, dt),
            dtype=float,
        )

        expected_shape = (
            self.state_dim,
            self.state_dim,
        )

        if Q.shape != expected_shape:
            raise ValueError(
                f"{self.name} process-noise covariance must have shape "
                f"{expected_shape}, but received {Q.shape}."
            )

        if not np.all(np.isfinite(Q)):
            raise ValueError(
                f"{self.name} process-noise covariance contains "
                "NaN or infinite values."
            )

        if not np.allclose(
            Q,
            Q.T,
            atol=1e-10,
        ):
            raise ValueError(
                f"{self.name} process-noise covariance must be symmetric."
            )

        Q = 0.5 * (Q + Q.T)

        if np.any(np.linalg.eigvalsh(Q) < -1e-10):
            raise ValueError(
                f"{self.name} process-noise covariance must be "
                "positive semidefinite."
            )

        return Q.copy()

    @abstractmethod
    def _process_noise(
        self,
        x: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """
        Construct Q(x, dt) using validated inputs.
        """
        raise NotImplementedError

    def jacobian(
        self,
        x: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """
        Return the transition Jacobian:

            F_k = df(x, dt) / dx

        EKF-compatible models must override this method.
        UKF-only models do not require it.
        """
        self.validate_state(x)
        self.validate_dt(dt)

        raise NotImplementedError(
            f"{self.name} does not provide a transition Jacobian."
        )

    def validate_jacobian(
        self,
        jacobian: np.ndarray,
    ) -> np.ndarray:
        """Validate and return a transition Jacobian."""
        F = np.asarray(
            jacobian,
            dtype=float,
        )

        expected_shape = (
            self.state_dim,
            self.state_dim,
        )

        if F.shape != expected_shape:
            raise ValueError(
                f"{self.name} transition Jacobian must have shape "
                f"{expected_shape}, but received {F.shape}."
            )

        if not np.all(np.isfinite(F)):
            raise ValueError(
                f"{self.name} transition Jacobian contains "
                "NaN or infinite values."
            )

        return F.copy()