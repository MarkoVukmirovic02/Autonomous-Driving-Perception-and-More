"""Shared contract for conventional measurement/observation models.

PURPOSE
-------
A motion model relates states at different times. A measurement model relates
one state to the observations used by the estimator at the same time.
Concrete models define the observation representation and its geometry.

For the usual additive model:
    Z = h(X) + V,
    E[V | X=x, specified context] = 0,
    Cov(V | X=x, specified context) = R.

Gaussian conditional noise gives a Gaussian observation law at fixed x;
nonlinear h does not generally give a Gaussian posterior or predictive
observation distribution when X is uncertain.

predict_measurement returns nominal h(x), not an observation sampled with
noise and not generally E[h(X)] for uncertain X. measurement_noise returns R,
not the full innovation covariance S, which the filter constructs using state
uncertainty as well.

This interface describes estimator inputs, which may be processed detections
rather than raw sensor outputs. It does not detect objects, associate returns,
manage timestamps, choose sensors or apply calibration automatically.
"""
from abc import ABC, abstractmethod

import numpy as np

from Perception.tracking.types import MeasurementSpaceIdentity


def _dimension(value, name):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))):
        raise TypeError(f"{name} must be an integer.")
    if value < 1:
        raise ValueError(f"{name} must be positive.")
    return int(value)


def _real_array(value, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in 'iuf':
        raise ValueError(f"{name} must contain real numerical values.")
    result = np.asarray(raw, dtype=float)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite.")
    return result.copy()


class MeasurementModel(ABC):
    """Sensor-independent contract for conventional observation models.

    Concrete models define name, state_dim, measurement_dim and, when semantic
    compatibility checks are used, measurement_space_identity. Annotations
    alone do not instantiate/validate metadata. Generic legacy numerical
    callers may omit an identity, but an explicit identity validation fails
    until one is supplied; identified IMM/association paths require it.

    angle_indices contains observation-coordinate indices measured in radians.
    Missing metadata defaults to no angles. It describes observation angles,
    not the state's heading indices. residual uses it to wrap differences.

    A state and its observation may use different coordinates and dimensions.
    Required sensor pose, calibration and ego motion must be incorporated by
    the concrete model through an explicit, documented configuration/context.
    A sensor identifier or matching frame name does not perform geometry.

    Abstract methods are an inheritance contract, not automatic wrappers:
    concrete methods must call the appropriate validation helpers. Filters
    retain numerical-boundary validation of returned model quantities too.
    """
    name: str
    state_dim: int
    measurement_dim: int
    measurement_space_identity: MeasurementSpaceIdentity

    def _measurement_angles(self):
        dimension = _dimension(self.measurement_dim, 'measurement_dim')
        indices = tuple(getattr(self, 'angle_indices', ()))
        if any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer))
               or i < 0 or i >= dimension for i in indices):
            raise ValueError('Observation angle indices must be in-bounds integers.')
        if len(set(indices)) != len(indices):
            raise ValueError('Observation angle indices must be unique.')
        return tuple(int(i) for i in indices)

    def validate_measurement_space_identity(self):
        """Require semantic identity to agree with model dimension/angle metadata.

        Identity distinguishes e.g. range/bearing/range-rate from Cartesian
        position even when both vectors have three entries. It specifies
        meanings/order, not a physical sensor or a particular coordinate frame.
        """
        _dimension(self.state_dim, 'state_dim')
        dimension = _dimension(self.measurement_dim, 'measurement_dim')
        identity = getattr(self, 'measurement_space_identity', None)
        if not isinstance(identity, MeasurementSpaceIdentity):
            raise TypeError('measurement model must declare a MeasurementSpaceIdentity')
        if (identity.dimension != dimension
                or identity.angle_indices != self._measurement_angles()):
            raise ValueError('measurement-space identity disagrees with dimension or angles')
        return identity

    def validate_state(self, x: np.ndarray) -> np.ndarray:
        """Return a detached finite real vector of shape (state_dim,).

        No silent flattening: row/column matrices require explicit conversion
        by the caller. Shape/finiteness checks do not establish physical bounds,
        observability or the correctness of a coordinate convention.
        """
        dimension = _dimension(self.state_dim, 'state_dim')
        state = _real_array(x, f'{self.name} state')
        if state.shape != (dimension,):
            raise ValueError(f'{self.name} expected state shape ({dimension},), got {state.shape}.')
        return state

    def validate_measurement(self, z: np.ndarray) -> np.ndarray:
        """Return a detached finite real vector of shape (measurement_dim,).

        This validates values rather than a Measurement metadata container.
        It does not wrap angular values or check frame, source or timestamps;
        adapters/trackers handle those policies at the appropriate boundary.
        """
        dimension = _dimension(self.measurement_dim, 'measurement_dim')
        measurement = _real_array(z, f'{self.name} measurement')
        if measurement.shape != (dimension,):
            raise ValueError(f'{self.name} expected measurement shape ({dimension},), got {measurement.shape}.')
        return measurement

    def validate_covariance(self, covariance: np.ndarray) -> np.ndarray:
        """Validate/detach an observation-noise covariance R.

        Accept PSD, including singular matrices. Tiny numerical asymmetry and
        negative eigenvalues within tolerance are corrected; materially invalid
        input is rejected. This is a numerical policy in the supplied coordinate
        units, not evidence that the measurement noise is calibrated.

        A model's measurement_noise implementation must call this helper to use
        it; this base cannot intercept an overridden abstract method. Filters may
        separately need positive-definite innovation covariance for their solves.
        """
        dimension = _dimension(self.measurement_dim, 'measurement_dim')
        R = _real_array(covariance, f'{self.name} measurement covariance')
        if R.shape != (dimension, dimension):
            raise ValueError(f'Measurement covariance must have shape ({dimension}, {dimension}).')
        if not np.allclose(R, R.T, rtol=0, atol=1e-10):
            raise ValueError('Measurement covariance must be symmetric.')
        R = .5*(R+R.T)
        values, vectors = np.linalg.eigh(R)
        tolerance = 1e-10*max(1., float(np.max(np.abs(values))))
        if values.min() < -tolerance:
            raise ValueError('Measurement covariance must be positive semidefinite.')
        if values.min() < 0:
            R = (vectors*np.maximum(values, 0)) @ vectors.T
            R = .5*(R+R.T)
        return R.copy()

    @abstractmethod
    def predict_measurement(self, x: np.ndarray) -> np.ndarray:
        """Return nominal h(x), shape (measurement_dim,), for a specified state.

        Concrete implementations validate the input and output. With zero
        conditional-mean noise h(x) is the conditional observation mean in
        Euclidean coordinates. Angular coordinates use a suitable local chart.
        """
        raise NotImplementedError

    @abstractmethod
    def measurement_noise(self) -> np.ndarray:
        """Return R, shape (measurement_dim, measurement_dim), in observation units.

        R must be finite, real, symmetric and PSD. Cross-coordinate entries can
        encode correlated errors; a diagonal covariance is not required.
        This no-argument interface supports configured noise, not explicit
        state-dependent R(x). Such a model requires a deliberate interface/filter
        extension. A per-observation R override can be supplied through Measurement
        when the consuming estimator supports it.
        """
        raise NotImplementedError

    def residual(self, z: np.ndarray, z_pred: np.ndarray) -> np.ndarray:
        """Return observation minus prediction, wrapping declared angles.

        Euclidean components use ordinary subtraction. Angular components use
        atan2(sin(delta),cos(delta)) for the shortest signed local difference;
        this avoids comparing bearings across the +/-pi cut as far apart.
        The antipodal difference has no unique direction. This local residual
        does not represent a general broad or multimodal circular likelihood.

        Specialized models may override this method if their observation
        geometry requires another residual. Residuals alone do not specify a
        likelihood: the uncertainty model and probabilistic assumptions matter.
        """
        measurement = self.validate_measurement(z)
        prediction = self.validate_measurement(z_pred)
        angles = self._measurement_angles()
        with np.errstate(over='raise', invalid='raise'):
            try:
                difference = measurement-prediction
            except FloatingPointError as error:
                raise ValueError('Observation residual exceeds numerical range.') from error
        if angles:
            indices = list(angles)
            difference[indices] = np.arctan2(np.sin(difference[indices]),
                                             np.cos(difference[indices]))
        return difference


# FUTURE OPERATOR-BASED EXTENSION
# -------------------------------
# A learned observation law estimates distributions of actual estimator inputs
# conditional on state/context. h and R need not describe that law completely.
# A learned likelihood evaluation interface may therefore be separate from this
# conventional moment/residual interface. Conditioning direction remains z|x;
# it is not the posterior x|z. The filter still combines likelihood and predicted
# belief through Bayes' rule. Training and evaluation must respect timestamps,
# geometry, association and any conditioning on successful detection.
