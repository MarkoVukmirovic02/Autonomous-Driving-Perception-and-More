from abc import ABC, abstractmethod

import numpy as np


class LinearMotionModel(ABC):
    """
    Base class for linear motion models.

    Linear state-transition equation:

        x_k = F(dt) x_{k-1} + w_k

    where:

        w_k ~ N(0, Q(dt))
    """

    name: str
    state_dim: int

    @staticmethod
    def validate_dt(dt: float) -> float:
        """
        Validate and return the prediction interval.
        """
        dt = float(dt)

        if not np.isfinite(dt):
            raise ValueError("dt must be finite.")

        if dt < 0.0:
            raise ValueError("dt must be non-negative.")

        return dt

    @abstractmethod
    def transition_matrix(
        self,
        dt: float,
    ) -> np.ndarray:
        """
        Return the state-transition matrix F(dt).
        """
        raise NotImplementedError

    @abstractmethod
    def process_noise(
        self,
        dt: float,
    ) -> np.ndarray:
        """
        Return the process-noise covariance Q(dt).
        """
        raise NotImplementedError