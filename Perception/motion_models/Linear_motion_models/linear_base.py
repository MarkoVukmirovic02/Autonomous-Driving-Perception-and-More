from abc import ABC, abstractmethod
import numpy as np


class LinearMotionModel(ABC):
    name: str
    state_dim: int

    @abstractmethod
    def transition_matrix(self, dt: float) -> np.ndarray:
        pass

    @abstractmethod
    def process_noise(self, dt: float) -> np.ndarray:
        pass