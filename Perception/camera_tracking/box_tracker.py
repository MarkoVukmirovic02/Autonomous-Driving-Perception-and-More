import numpy as np

from Perception.filters.kalman_filter import KalmanFilter
from Perception.motion_models.linear.cv import ConstantVelocityModel
from Perception.measurement_models.linear.yolo_box import (
    BoundingBoxMeasurementModel,
)


class BoxKalmanFilter:
    """
    Thin adapter between YOLO bounding boxes and the generic Kalman filter.

    Hidden state:

        [cx, cy, width, height,
         cx_rate, cy_rate, width_rate, height_rate]
    """

    def __init__(
        self,
        bbox,
        acceleration_std: float | np.ndarray = None,
        corner_covariance: np.ndarray = None,
    ) -> None:
        if acceleration_std is None:
            acceleration_std = np.array(
                [20.0, 20.0, 10.0, 10.0],
                dtype=float,
            )

        if corner_covariance is None:
            edge_std = 3.0

            corner_covariance = (
                edge_std**2 * np.eye(4, dtype=float)
            )

        self.motion_model = ConstantVelocityModel(
            dimensions=4,
            acceleration_std=acceleration_std,
        )

        self.measurement_model = BoundingBoxMeasurementModel(
            state_dim=self.motion_model.state_dim,
            corner_covariance=corner_covariance,
        )

        # Converts [x1, y1, x2, y2] into [cx, cy, width, height].
        z0 = self.measurement_model.raw_to_measurement(
            bbox
        )

        x0 = np.concatenate(
            [
                z0,
                np.zeros(4, dtype=float),
            ]
        )

        P0 = np.diag(
            [
                10.0,
                10.0,
                10.0,
                10.0,
                100.0,
                100.0,
                100.0,
                100.0,
            ]
        )

        self.kf = KalmanFilter(
            x0=x0,
            P0=P0,
            motion_model=self.motion_model,
            measurement_model=self.measurement_model,
        )

    def predict(
        self,
        dt: float,
    ) -> np.ndarray:
        return self.kf.predict(dt)

    def update(
        self,
        bbox,
    ) -> np.ndarray:
        z = self.measurement_model.raw_to_measurement(
            bbox
        )

        return self.kf.update(z)

    def step(
        self,
        bbox,
        dt: float,
    ) -> np.ndarray:
        z = self.measurement_model.raw_to_measurement(
            bbox
        )

        return self.kf.step(
            z=z,
            dt=dt,
        )

    def predict_k_steps(
        self,
        k: int,
        dt: float,
    ) -> np.ndarray:
        """
        Project the current state mean k steps into the future.

        This method does not modify the filter and does not propagate
        covariance. It is intended for visualization and approximate
        future bounding-box prediction.
        """
        if not isinstance(k, int):
            raise TypeError("k must be an integer.")

        if k < 0:
            raise ValueError("k must be non-negative.")

        future_state = self.kf.x.copy()

        F = self.motion_model.transition_matrix(dt)

        for _ in range(k):
            future_state = F @ future_state

        return future_state