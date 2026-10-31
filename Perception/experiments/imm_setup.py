# Perception/experiments/imm_setup.py

import numpy as np

from Perception.filters.kalman_filter import KalmanFilter
from Perception.filters.extended_kalman_filter import ExtendedKalmanFilter
from Perception.filters.unscented_kalman_filter import UnscentedKalmanFilter

from Perception.motion_models.linear.cv import ConstantVelocityModel
from Perception.motion_models.linear.ca import ConstantAccelerationModel
from Perception.motion_models.nonlinear.ctrv import CTRVModel
from Perception.motion_models.nonlinear.ctra import CTRAModel

from Perception.measurement_models.linear.linear_base import (
    LinearMeasurementModel,
)

from Perception.tracking.types import MeasurementSpaceIdentity
from Perception.tracking.imm import IMM
from Perception.tracking.imm_adapter import (
    make_cv_adapter,
    make_ca_adapter,
    make_ctrv_adapter,
    make_ctra_adapter,
)
from Perception.tracking.state_conversion import (
    common_to_cv_mean,
    common_to_cv_covariance,
    common_to_ca_mean,
    common_to_ca_covariance,
    common_to_ctrv_mean,
    common_to_ctrv_covariance,
    common_to_ctra_mean,
    common_to_ctra_covariance,
)


class PositionMeasurementModel(LinearMeasurementModel):
    """Experiment observation: Cartesian position in the state frame."""

    name = "Planar position"
    measurement_dim = 2
    angle_indices = ()

    measurement_space_identity = MeasurementSpaceIdentity(
        name="planar_position",
        coordinates=("position_x_m", "position_y_m"),
        dimension=2,
    )

    def __init__(self, state_dim, covariance):
        self.state_dim = state_dim
        self._R = np.array(covariance, dtype=float, copy=True)

        self._H = np.zeros((2, state_dim))
        self._H[:, :2] = np.eye(2)

    def measurement_matrix(self):
        return self._H.copy()

    def measurement_noise(self):
        return self._R.copy()


def build_imm():
    # Common initial belief:
    # [px, py, vx, vy, ax, ay]
    #
    # An illustrative moving-object prior, not an initialization
    # procedure for an arbitrary detection.
    common_mean = np.array([0., 0., 8., 0., 0., 0.])
    common_covariance = np.diag([
        0.7**2, 0.7**2,
        1.0**2, 1.0**2,
        1.0**2, 1.0**2,
    ])

    R = np.diag([0.7**2, 0.7**2])

    # Reverse conversion currently rejects speeds below this threshold.
    minimum_speed = 0.5  # m/s

    # Native initial distributions, derived from one common belief.
    cv_x = common_to_cv_mean(common_mean)
    cv_P = common_to_cv_covariance(common_covariance)

    ca_x = common_to_ca_mean(common_mean)
    ca_P = common_to_ca_covariance(common_covariance)

    ctrv_x = common_to_ctrv_mean(
        common_mean, minimum_speed=minimum_speed
    )
    ctrv_P = common_to_ctrv_covariance(
        common_mean,
        common_covariance,
        minimum_speed=minimum_speed,
    )

    ctra_x = common_to_ctra_mean(
        common_mean, minimum_speed=minimum_speed
    )
    ctra_P = common_to_ctra_covariance(
        common_mean,
        common_covariance,
        minimum_speed=minimum_speed,
    )

    # Existing motion-model implementations.
    # These noise settings are illustrative experiment parameters.
    cv_motion = ConstantVelocityModel(
        dimensions=2,
        acceleration_std=0.5,
    )
    ca_motion = ConstantAccelerationModel(
        dimensions=2,
        jerk_std=0.7,
    )
    ctrv_motion = CTRVModel(
        acceleration_std=0.7,
        yaw_acceleration_std=0.15,
    )
    ctra_motion = CTRAModel(
        jerk_std=0.9,
        yaw_acceleration_std=0.15,
    )

    # Existing filter implementations.
    cv_filter = KalmanFilter(
        x0=cv_x,
        P0=cv_P,
        motion_model=cv_motion,
        measurement_model=PositionMeasurementModel(4, R),
    )

    ca_filter = KalmanFilter(
        x0=ca_x,
        P0=ca_P,
        motion_model=ca_motion,
        measurement_model=PositionMeasurementModel(6, R),
    )

    ctrv_filter = ExtendedKalmanFilter(
        x0=ctrv_x,
        P0=ctrv_P,
        motion_model=ctrv_motion,
        measurement_model=PositionMeasurementModel(5, R),
        state_angle_indices=(3,),
    )

    ctra_filter = UnscentedKalmanFilter(
        x0=ctra_x,
        P0=ctra_P,
        motion_model=ctra_motion,
        measurement_model=PositionMeasurementModel(6, R),
        state_angle_indices=(4,),
    )

    # Conversion priors are separate from process-noise parameters.
    adapters = [
        make_cv_adapter(
            cv_filter,
            coordinate_frame="world",
            acceleration_covariance=np.diag([0.8**2, 0.8**2]),
        ),
        make_ca_adapter(ca_filter, coordinate_frame="world"),
        make_ctrv_adapter(
            ctrv_filter,
            coordinate_frame="world",
            longitudinal_acceleration_variance=0.8**2,
            minimum_speed=minimum_speed,
        ),
        make_ctra_adapter(
            ctra_filter,
            coordinate_frame="world",
            minimum_speed=minimum_speed,
        ),
    ]

    stay_probability = 0.90
    transition_matrix = np.full(
        (4, 4), (1.0 - stay_probability) / 3.0
    )
    np.fill_diagonal(transition_matrix, stay_probability)

    return IMM(
        filters=adapters,
        mode_probabilities=np.full(4, 0.25),
        transition_matrix=transition_matrix,
    )


def main():
    imm = build_imm()

    # A short execution check with one noisy position measurement.
    dt = 0.2
    rng = np.random.default_rng(42)

    true_position = np.array([8.0 * dt, 0.0])
    measurement = true_position + rng.normal(0.0, 0.7, size=2)

    imm.predict(dt)
    imm.update(measurement)

    mean, covariance = imm.combine_estimates()

    print("Combined state:", mean)
    print("Mode probabilities:", imm.mode_probabilities)
    print("Covariance diagonal:", np.diag(covariance))


if __name__ == "__main__":
    main()