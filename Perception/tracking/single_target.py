"""Estimator interfaces and time coordination for one tracked object.

`types.py` defines the values exchanged between components. This module
specifies the estimator operations used by association, track management
and the single-object wrapper. Filter equations remain in the estimators.

A Protocol describes structural compatibility: implementations need not
inherit from it explicitly. runtime_checkable permits isinstance checks,
but does not verify signatures, numerical validity or mathematical behavior.

This module performs no sensor decoding or coordinate transformations.
"""
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from .types import (GaussianState, InnovationDiagnostics, Measurement,
                    MeasurementPrediction, MeasurementMoments, MeasurementSpaceIdentity, StateEstimate,
                    validate_gaussian_state)


@runtime_checkable
class AssociationEstimator(Protocol):
    """Observation-compatibility operations used before association.

    Prediction/diagnostics methods must not change the estimator belief.
    Here x means the estimated mean; P is its covariance. These Gaussian
    summaries do not describe a general non-Gaussian belief completely.
    """
    x: np.ndarray
    P: np.ndarray
    measurement_angle_indices: tuple[int, ...]

    @property
    def measurement_space_identity(self) -> MeasurementSpaceIdentity | None: ...

    def measurement_prediction(self, R=None) -> MeasurementPrediction | MeasurementMoments: ...
    def innovation_diagnostics(self, z, R=None) -> InnovationDiagnostics: ...
    def state_snapshot(self) -> GaussianState: ...


@runtime_checkable
class ManagedEstimator(AssociationEstimator, Protocol):
    """Operations generic multi-object management needs from each track.

    Management decides which measurement belongs to a track; the estimator
    performs prediction and the assigned measurement update. Return values
    are intentionally unspecified because existing estimators differ.
    """
    def predict(self, dt: float): ...
    def update_measurement(self, measurement: Measurement): ...


@runtime_checkable
class SingleTargetEstimator(AssociationEstimator, Protocol):
    """Numerical operations needed by the single-object time/frame wrapper.

    A bank estimator need not own a single motion/observation model or support
    replacing its complete belief with one Gaussian.
    """
    state_angle_indices: tuple[int, ...]

    def predict(self, dt: float): ...
    def update(self, z, R=None): ...


@runtime_checkable
class StateReplacementEstimator(Protocol):
    """Optional native Gaussian replacement capability, used by raw filters."""
    state_angle_indices: tuple[int, ...]
    def replace_state(self, x, P) -> GaussianState: ...


def _identifier(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _nonnegative(value, name):
    # Explicitly reject booleans and complex values rather than interpreting
    # True as one second or silently discarding an imaginary component.
    raw = np.asarray(value)
    if raw.ndim != 0 or raw.dtype.kind not in 'iuf' or np.iscomplexobj(raw):
        raise ValueError(f"{name} must be a real numerical scalar")
    result = float(raw)
    if not np.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


@dataclass(frozen=True)
class SingleTargetCheckpoint:
    """Estimator checkpoint plus this wrapper's clock and declared metadata."""
    estimator_checkpoint: object
    owner_id: int
    timestamp_s: float | None
    last_dt_s: float | None
    coordinate_frame: str
    measurement_frame: str
    sensor_id: str | None


class SingleTargetTracker:
    """Coordinate one estimator's prediction/update with time and frames.

    coordinate_frame describes the estimated STATE, not necessarily the
    observations. measurement_frame describes the expected observation frame;
    it defaults to coordinate_frame for compatibility with existing callers.
    The observation model must implement the geometry relating these frames.
    Setting different names does not itself implement that geometry.

    sensor_id optionally restricts this wrapper to one declared sensor.
    When configured, measurements must carry that exact sensor_id. Otherwise
    source metadata is accepted but does not automatically select a model.

    Missing measurement time/frame/identity remain allowed for existing
    manually coordinated callers. They mean unspecified, not verified.
    Timestamped operation should supply an initial timestamp and capture times.

    Validation of supplied metadata precedes mutation. Arbitrary failures
    inside estimator.predict/update are NOT transactional: this wrapper cannot
    restore all internal estimator caches through a Gaussian snapshot alone.
    A numerical failure therefore requires inspecting/restoring the estimator
    before retrying; a failed update after prediction leaves the predicted time.
    """

    def __init__(self, estimator: SingleTargetEstimator, *, coordinate_frame: str,
                 initial_timestamp_s: float | None = None,
                 measurement_frame: str | None = None,
                 sensor_id: str | None = None):
        if not isinstance(estimator, SingleTargetEstimator):
            raise TypeError("estimator must satisfy SingleTargetEstimator")
        self.coordinate_frame = _identifier(coordinate_frame, "state coordinate frame")
        self.measurement_frame = _identifier(
            coordinate_frame if measurement_frame is None else measurement_frame,
            "measurement frame")
        self.sensor_id = None if sensor_id is None else _identifier(sensor_id, "sensor_id")
        # Reject an invalid initial belief before making the wrapper available.
        initial = estimator.state_snapshot()
        if not isinstance(initial, GaussianState):
            raise TypeError("state_snapshot must return GaussianState")
        declared_state_frame = getattr(estimator, 'coordinate_frame', None)
        declared_measurement_frame = getattr(estimator, 'measurement_frame', None)
        if declared_state_frame is not None and declared_state_frame != self.coordinate_frame:
            raise ValueError("estimator state coordinate-frame mismatch")
        if declared_measurement_frame is not None and declared_measurement_frame != self.measurement_frame:
            raise ValueError("estimator measurement coordinate-frame mismatch")
        self.estimator = estimator
        self.last_timestamp_s = (None if initial_timestamp_s is None
                                 else self._timestamp(initial_timestamp_s))
        self.last_dt_s = None

    @staticmethod
    def _timestamp(value):
        return _nonnegative(value, "timestamp")

    @property
    def state(self):
        """Detached output-state mean; modifying it does not update the filter."""
        return self.estimator.state_snapshot().x.copy()

    @property
    def x(self):
        return self.state

    @property
    def covariance(self):
        """Detached covariance in the estimator's output state coordinates."""
        return self.estimator.state_snapshot().P.copy()

    @property
    def P(self):
        return self.covariance

    def snapshot(self):
        """Export one coherent output-state snapshot with state-frame metadata."""
        state = self.estimator.state_snapshot()
        return StateEstimate(state.x, state.P, self.last_timestamp_s,
                             self.coordinate_frame)

    def checkpoint(self):
        """Capture estimator and clock together when supported (for example IMM)."""
        capture = getattr(self.estimator, 'checkpoint', None)
        if not callable(capture):
            raise TypeError("estimator does not support complete bank checkpoints")
        return SingleTargetCheckpoint(capture(), id(self), self.last_timestamp_s,
                                      self.last_dt_s, self.coordinate_frame,
                                      self.measurement_frame, self.sensor_id)

    def restore_checkpoint(self, checkpoint):
        """Restore a checkpoint from this wrapper; external sensor context is excluded."""
        if not isinstance(checkpoint, SingleTargetCheckpoint):
            raise TypeError("expected SingleTargetCheckpoint")
        if (checkpoint.owner_id != id(self)
                or checkpoint.coordinate_frame != self.coordinate_frame
                or checkpoint.measurement_frame != self.measurement_frame
                or checkpoint.sensor_id != self.sensor_id):
            raise ValueError("checkpoint belongs to a different wrapper/configuration")
        restore = getattr(self.estimator, 'restore_checkpoint', None)
        if not callable(restore):
            raise TypeError("estimator does not support checkpoint restoration")
        restore(checkpoint.estimator_checkpoint)
        self.last_timestamp_s = checkpoint.timestamp_s
        self.last_dt_s = checkpoint.last_dt_s
        return self.snapshot()

    def replace_state(self, x, P):
        """Replace the belief at its existing time; delegate cache clearing."""
        if not isinstance(self.estimator, StateReplacementEstimator):
            raise TypeError("estimator does not support Gaussian state replacement; "
                            "restore an IMM bank checkpoint instead")
        current = self.estimator.state_snapshot()
        validated = validate_gaussian_state(
            x, P, current.x.size, self.estimator.state_angle_indices)
        self.estimator.replace_state(validated.x, validated.P)
        return self.snapshot()

    def _validate_frame(self, coordinate_frame):
        # predict/process coordinate_frame arguments refer to the STATE frame.
        if coordinate_frame is not None:
            _identifier(coordinate_frame, "state coordinate frame")
            if coordinate_frame != self.coordinate_frame:
                raise ValueError("state coordinate-frame mismatch")

    def _validate_measurement_identity(self, measurement):
        if not isinstance(measurement, Measurement):
            raise TypeError("expected Measurement")
        declared = getattr(self.estimator, 'measurement_frame', None)
        if declared is not None and declared != self.measurement_frame:
            raise ValueError("estimator measurement coordinate-frame mismatch")
        if (measurement.coordinate_frame is not None
                and measurement.coordinate_frame != self.measurement_frame):
            raise ValueError("measurement coordinate-frame mismatch")
        if self.sensor_id is not None and measurement.sensor_id != self.sensor_id:
            raise ValueError("measurement sensor_id does not match tracker")
        expected = getattr(self.estimator, 'measurement_space_identity', None)
        identity = measurement.measurement_space_identity
        if identity is not None and identity != expected:
            raise ValueError("measurement space does not match estimator")
        # Even an observation lacking an identity must match the expected size
        # when the estimator declares one. Never infer semantics from size alone.
        if expected is not None:
            if not isinstance(expected, MeasurementSpaceIdentity):
                raise TypeError("estimator measurement identity is invalid")
            if measurement.value.shape != (expected.dimension,):
                raise ValueError("measurement dimension does not match estimator")

    @staticmethod
    def _validate_measurement_time(measurement, target):
        if (measurement.timestamp_s is not None and target is not None
                and not np.isclose(measurement.timestamp_s, target,
                                   rtol=0, atol=1e-12)):
            raise ValueError("measurement timestamp does not match predicted timestamp")

    def _prediction_schedule(self, dt, timestamp_s, coordinate_frame):
        """Calculate the target time without advancing the estimator."""
        self._validate_frame(coordinate_frame)
        declared = getattr(self.estimator, 'coordinate_frame', None)
        if declared is not None and declared != self.coordinate_frame:
            raise ValueError("estimator state coordinate-frame mismatch")
        if (dt is None) == (timestamp_s is None):
            raise ValueError("provide exactly one of dt or timestamp_s")
        if timestamp_s is not None:
            timestamp = self._timestamp(timestamp_s)
            if self.last_timestamp_s is None:
                raise ValueError("timestamp prediction requires an initial timestamp")
            interval = timestamp - self.last_timestamp_s
            if interval <= 0:
                raise ValueError("timestamps must strictly increase")
        else:
            interval = _nonnegative(dt, "dt")
            timestamp = (None if self.last_timestamp_s is None
                         else self._timestamp(self.last_timestamp_s + interval))
        return interval, timestamp

    def _apply_prediction(self, interval, timestamp):
        self.estimator.predict(interval)
        # Commit clock metadata only after predict returns successfully.
        self.last_dt_s = interval
        self.last_timestamp_s = timestamp

    def predict(self, *, dt: float | None = None, timestamp_s: float | None = None,
                coordinate_frame: str | None = None):
        """Advance using exactly one of an interval or absolute target time.

        Absolute times must increase. dt=0 is allowed for explicit same-time
        operation. This wrapper does not buffer/reorder delayed observations.
        """
        interval, timestamp = self._prediction_schedule(dt, timestamp_s, coordinate_frame)
        self._apply_prediction(interval, timestamp)
        return self.snapshot()

    def update(self, measurement: Measurement):
        """Condition the current belief without advancing its time.

        Multiple updates at one time are possible, but independence/correlation
        assumptions must be handled by the observation models and caller.
        A timestamp alone cannot establish the time of an unanchored belief.
        """
        self._validate_measurement_identity(measurement)
        self._validate_measurement_time(measurement, self.last_timestamp_s)
        self.estimator.update(measurement.value, R=measurement.covariance)
        return self.snapshot()

    def measurement_prediction(self, R=None):
        """Inspect predicted observation moments at the CURRENT belief time."""
        return self.estimator.measurement_prediction(R=R)

    def innovation_diagnostics(self, measurement: Measurement):
        """Inspect compatibility at the current time without prediction/update."""
        self._validate_measurement_identity(measurement)
        self._validate_measurement_time(measurement, self.last_timestamp_s)
        return self.estimator.innovation_diagnostics(
            measurement.value, R=measurement.covariance)

    def process(self, measurement: Measurement | None = None, *, dt: float | None = None,
                timestamp_s: float | None = None, coordinate_frame: str | None = None):
        """Validate the schedule, predict, then optionally incorporate a measurement.

        If no explicit schedule is supplied, use the measurement capture time.
        Metadata conflicts are rejected BEFORE prediction. Passing None skips
        updating; this does not model information carried by a missed detection.
        """
        if measurement is not None:
            self._validate_measurement_identity(measurement)
            if timestamp_s is None and dt is None:
                timestamp_s = measurement.timestamp_s
        interval, target = self._prediction_schedule(dt, timestamp_s, coordinate_frame)
        if measurement is not None:
            self._validate_measurement_time(measurement, target)
        self._apply_prediction(interval, target)
        return self.snapshot() if measurement is None else self.update(measurement)


# FUTURE OPERATOR-BASED EXTENSION
# -------------------------------
# Time, sensor provenance and frame checks remain useful when the belief is
# represented by support weights or feature coefficients. GaussianState, x/P
# snapshots and Gaussian innovation diagnostics are not a universal contract
# for that belief. Introduce a separate belief interface once its representation
# is chosen; do not force a rich distribution into mean/covariance just to fit
# these protocols. Association may then use predictive observation probabilities
# from that representation rather than a single-Gaussian approximation.
# Learned transition/observation models still require the same timing and
# geometry conventions. This wrapper does not implement those learned models.
