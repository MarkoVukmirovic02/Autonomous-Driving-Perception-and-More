from abc import ABC, abstractmethod

import numpy as np



class NonlinearMotionModel(ABC):
    name: str
    state_dim: int

    @staticmethod
    def normalize_angle(angle: float) -> float:
        """Normalize an angle to [-pi, pi)."""
        return (angle + np.pi) % (2.0 * np.pi) - np.pi

    def validate_state(self, x: np.ndarray) -> np.ndarray:
        """Return a validated one-dimensional state vector."""
        state = np.asarray(x, dtype=float).reshape(-1)

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

    @abstractmethod
    def predict_state(
        self,
        x: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def process_noise(
        self,
        x: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        raise NotImplementedError

    def jacobian(
        self,
        x: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        raise NotImplementedError(
            f"{self.name} does not provide a transition Jacobian."
        )