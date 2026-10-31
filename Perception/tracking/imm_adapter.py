import numpy as np
from .types import GaussianState
from .state_conversion import (
    cv_to_common_mean,
    cv_to_common_covariance,
    common_to_cv_mean,
    common_to_cv_covariance,
    ca_to_common_mean,
    ca_to_common_covariance,
    common_to_ca_mean,
    common_to_ca_covariance,
    ctrv_to_common_mean,
    ctrv_to_common_covariance,
    common_to_ctrv_mean,
    common_to_ctrv_covariance,
    ctra_to_common_mean,
    ctra_to_common_covariance,
    common_to_ctra_mean,
    common_to_ctra_covariance,
    common_to_ctrv_low_speed,
    common_to_ctra_low_speed,
)
PLANAR_COMMON_SPACE = ("px_m", "py_m", "vx_m_s", "vy_m_s", "ax_m_s2", "ay_m_s2")


class IMMFilterAdapter:
    """Native filter plus explicit conversion/coordinate metadata.

    Unspecified frames (None) preserve manually coordinated legacy callers.
    For deployment, supply frames to every factory and use SingleTargetTracker.
    Custom mappings must describe one common Euclidean state with matching units.
    """
    def __init__(
        self,
        filter_,
        *,
        to_common,
        to_native,
        state_space,
        low_speed_conversion=None,
        minimum_speed=None,
        common_space=None,
        coordinate_frame=None,
        measurement_frame=None,
    ):
        self.filter = filter_
        self.to_common = to_common
        self.to_native = to_native
        self.state_space = state_space
        self.common_space = common_space
        for name, value in (("coordinate_frame", coordinate_frame),
                            ("measurement_frame", measurement_frame)):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be a nonempty string or None")
        self.coordinate_frame = coordinate_frame
        model_frame = getattr(getattr(filter_, 'measurement_model', None), 'frame_id', None)
        if measurement_frame is not None and model_frame is not None and measurement_frame != model_frame:
            raise ValueError("adapter measurement frame disagrees with observation model")
        self.measurement_frame = (measurement_frame if measurement_frame is not None
                                  else model_frame if model_frame is not None else coordinate_frame)
        if not callable(to_common) or not callable(to_native):
            raise TypeError("conversion functions must be callable")
        if low_speed_conversion is not None and not callable(low_speed_conversion):
            raise TypeError("low_speed_conversion must be callable")

        if low_speed_conversion is not None:
            if minimum_speed is None:
                raise ValueError(
                    "Low-speed conversion requires minimum_speed."
                )

            minimum_speed = float(minimum_speed)

            if not np.isfinite(minimum_speed) or minimum_speed <= 0:
                raise ValueError(
                    "minimum_speed must be finite and positive."
                )

        self.low_speed_conversion = low_speed_conversion
        self.minimum_speed = minimum_speed

    @property
    def measurement_context(self):
        """Optional model-declared geometry key; not inferred from vector size."""
        return getattr(getattr(self.filter, 'measurement_model', None),
                       'measurement_context', None)

    @property
    def measurement_space_identity(self):
        return self.filter.measurement_space_identity

    @property
    def measurement_angle_indices(self):
        return tuple(self.filter.measurement_angle_indices)

    def measurement_prediction(self, R=None):
        return self.filter.measurement_prediction(R=R)

    @property
    def x(self):
        return self.filter.state_snapshot().x

    @property
    def P(self):
        return self.filter.state_snapshot().P

    @property
    def state_angle_indices(self):
        return tuple(self.filter.state_angle_indices)

    def state_snapshot(self):
        return self.filter.state_snapshot()

    def common_state(self):
        native = self.state_snapshot()
        return self.to_common(native.x, native.P)

    def convert_from(
        self,
        source,
        source_state,
        destination_state=None,
    ):
        if source.common_space != self.common_space:
            raise ValueError("common state spaces do not match")
        if source.coordinate_frame != self.coordinate_frame:
            raise ValueError("source and destination state frames do not match")
        if source.state_space == self.state_space:
            return GaussianState(source_state.x, source_state.P)

        common = source.to_common(
            source_state.x,
            source_state.P,
        )

        if self.low_speed_conversion is not None:
            speed = np.hypot(common.x[2], common.x[3])

            if speed < self.minimum_speed:
                if destination_state is None:
                    raise ValueError(
                        "Low-speed conversion requires a "
                        "destination snapshot."
                    )

                return self.low_speed_conversion(
                    common.x,
                    common.P,
                    destination_state,
                )

        return self.to_native(common.x, common.P)

    def replace_state(self, x, P):
        """Install a distribution already in native coordinates."""
        return self.filter.replace_state(x, P)

    def predict(self, dt):
        return self.filter.predict(dt)

    def innovation_diagnostics(self, z, R=None):
        return self.filter.innovation_diagnostics(z, R=R)

    def update(self, z, R=None):
        return self.filter.update(z, R=R)


def make_cv_adapter(filter_, *, acceleration_covariance, coordinate_frame=None, measurement_frame=None):
    acceleration_covariance = np.array(
        acceleration_covariance, dtype=float, copy=True
    )

    def to_common(x, P):
        return GaussianState(
            cv_to_common_mean(x),
            cv_to_common_covariance(P, acceleration_covariance),
        )

    def to_native(x, P):
        return GaussianState(
            common_to_cv_mean(x),
            common_to_cv_covariance(P),
        )

    return IMMFilterAdapter(
        filter_,
        to_common=to_common,
        to_native=to_native,
        common_space=PLANAR_COMMON_SPACE,
        coordinate_frame=coordinate_frame,
        measurement_frame=measurement_frame,
        state_space=("planar_cv", "px", "py", "vx", "vy"),
    )


def make_ca_adapter(filter_, *, coordinate_frame=None, measurement_frame=None):
    def to_common(x, P):
        return GaussianState(
            ca_to_common_mean(x),
            ca_to_common_covariance(P),
        )

    def to_native(x, P):
        return GaussianState(
            common_to_ca_mean(x),
            common_to_ca_covariance(P),
        )

    return IMMFilterAdapter(
        filter_,
        to_common=to_common,
        to_native=to_native,
        common_space=PLANAR_COMMON_SPACE,
        coordinate_frame=coordinate_frame,
        measurement_frame=measurement_frame,
        state_space=("planar_ca", "px", "py", "vx", "vy", "ax", "ay"),
    )

def make_ctrv_adapter(
    filter_,
    *,
    longitudinal_acceleration_variance,
    minimum_speed,
    coordinate_frame=None,
    measurement_frame=None,
):
    variance = float(longitudinal_acceleration_variance)
    minimum_speed = float(minimum_speed)

    if not np.isfinite(variance) or variance < 0:
        raise ValueError(
            "Longitudinal acceleration variance must be "
            "finite and nonnegative."
        )

    if not np.isfinite(minimum_speed) or minimum_speed <= 0:
        raise ValueError(
            "minimum_speed must be finite and positive."
        )

    def to_common(x, P):
        return GaussianState(
            ctrv_to_common_mean(x),
            ctrv_to_common_covariance(x, P, variance),
        )

    def to_native(x, P):
        return GaussianState(
            common_to_ctrv_mean(
                x, minimum_speed=minimum_speed
            ),
            common_to_ctrv_covariance(
                x, P, minimum_speed=minimum_speed
            ),
        )

    return IMMFilterAdapter(
        filter_,
        to_common=to_common,
        to_native=to_native,
        common_space=PLANAR_COMMON_SPACE,
        coordinate_frame=coordinate_frame,
        measurement_frame=measurement_frame,
        state_space=(
            "planar_ctrv", "px", "py", "v", "yaw", "yaw_rate"
        ),
        low_speed_conversion=common_to_ctrv_low_speed,
        minimum_speed=minimum_speed,
    )

def make_ctra_adapter(filter_, *, minimum_speed, coordinate_frame=None, measurement_frame=None):
    minimum_speed = float(minimum_speed)

    if not np.isfinite(minimum_speed) or minimum_speed <= 0:
        raise ValueError(
            "minimum_speed must be finite and positive."
        )

    def to_common(x, P):
        return GaussianState(
            ctra_to_common_mean(x),
            ctra_to_common_covariance(x, P),
        )

    def to_native(x, P):
        return GaussianState(
            common_to_ctra_mean(
                x, minimum_speed=minimum_speed
            ),
            common_to_ctra_covariance(
                x, P, minimum_speed=minimum_speed
            ),
        )

    return IMMFilterAdapter(
        filter_,
        to_common=to_common,
        to_native=to_native,
        common_space=PLANAR_COMMON_SPACE,
        coordinate_frame=coordinate_frame,
        measurement_frame=measurement_frame,
        state_space=(
            "planar_ctra", "px", "py", "v", "a", "yaw", "yaw_rate"
        ),
        low_speed_conversion=common_to_ctra_low_speed,
        minimum_speed=minimum_speed,
    )
