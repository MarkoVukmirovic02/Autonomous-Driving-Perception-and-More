"""Shared data containers for conventional estimators and trackers.

PURPOSE
-------
This module gives filters, sensor adapters, IMM and multi-target tracking
a common vocabulary for exchanging numerical results.

It contains data structures and a Gaussian-state validation function.
It does not implement prediction, Bayesian updating, association or
coordinate transformations.

Examples of information exchanged through these types:
    * A sensor adapter supplies an observation as a Measurement.
    * A filter exposes its estimated mean/covariance as a GaussianState.
    * A filter predicts observation moments as a MeasurementPrediction.
    * Association code examines InnovationDiagnostics.
    * A tracker exposes a timestamped StateEstimate.

MATHEMATICAL CONVENTIONS
-----------------------
In the thesis, x_k denotes the modeled physical state.

In GaussianState and StateEstimate, the field named `x` instead stores
the estimated state mean. It is an implementation naming convention,
not a claim that the estimated mean equals the true physical state.

A Gaussian belief is completely determined by its mean and covariance.
For an approximate Gaussian filter, these parameters describe the
filter's Gaussian approximation rather than necessarily the exact
Bayesian posterior.

IMPORTANT PYTHON DETAIL
-----------------------
`@dataclass(frozen=True)` prevents ordinary reassignment of fields:

    state.x = another_array

It does not automatically prevent modifying an array stored in a field:

    state.x[0] = 100.0

Copying an array prevents sharing its storage with the caller.
All numerical result containers validate and detach their arrays.
Measurement arrays are read-only. Prediction and diagnostic arrays preserve
read-only input flags; GaussianState and StateEstimate arrays remain writable.
"""

from dataclasses import dataclass

import numpy as np

# Shared validation keeps every container consistent. Arrays are detached but
# remain writable unless a container explicitly marks them read-only.
def _dimension(value, name="dimension"):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return int(value)


def _identifier(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _real_array(value, name):
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real")
    result = np.asarray(raw, dtype=float)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result.copy()


def _vector(value, name):
    result = _real_array(value, name)
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{name} must be a nonempty vector")
    return result


def _angles(values, dimension):
    result = tuple(values)
    if any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer))
           or i < 0 or i >= dimension for i in result):
        raise ValueError("angle indices must be integers within the vector bounds")
    if len(set(result)) != len(result):
        raise ValueError("angle indices must be unique")
    return tuple(int(i) for i in result)


def _covariance(value, dimension, name):
    result = _real_array(value, name)
    if result.shape != (dimension, dimension):
        raise ValueError(f"{name} must have shape ({dimension}, {dimension})")
    # Tolerances are numerical policies in the supplied coordinate units.
    if not np.allclose(result, result.T, rtol=0, atol=1e-10):
        raise ValueError(f"{name} must be symmetric")
    result = (result + result.T) / 2
    values, vectors = np.linalg.eigh(result)
    tolerance = 1e-10 * max(1., float(np.max(np.abs(values))))
    if values.min() < -tolerance:
        raise ValueError(f"{name} must be positive semidefinite")
    if values.min() < 0:
        result = (vectors * np.maximum(values, 0)) @ vectors.T
        result = (result + result.T) / 2
    return result


def _timestamp(value):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or np.iscomplexobj(value):
        raise ValueError("timestamp must be real, finite and non-negative")
    result = float(value)
    if not np.isfinite(result) or result < 0:
        raise ValueError("timestamp must be finite and non-negative")
    return result



@dataclass(frozen=True)
class MeasurementSpaceIdentity:
    """Describe the meaning and ordering of a measurement vector.

    Shape alone is insufficient to establish measurement compatibility.
    For example, both of the following have dimension three:

        [range, azimuth, radial_velocity]
        [Cartesian_x, Cartesian_y, Cartesian_z]

    They require different observation models.

    This identity allows consuming code to check that a measurement
    matches the observation coordinates expected by an estimator.

    Attributes
    ----------
    name:
        Descriptive identifier for the measurement representation.

    coordinates:
        Ordered coordinate names. Names should document units where
        useful, for example:

            ("range_m", "azimuth_rad", "radial_velocity_m_s")

        Their order must agree with the numerical observation vector.

    dimension:
        Number of coordinates in the observation vector.

    angle_indices:
        Zero-based indices of periodic angular coordinates.

        For [range, azimuth, radial_velocity], this would be (1,).
        Consuming code uses this metadata when computing wrapped
        residuals or angular means.

    Scope
    -----
    This describes measurement semantics, not a particular sensor.
    Two radar sensors may share this identity while having different
    poses, timestamps and noise covariances.

    It does not perform unit conversion, coordinate transformations,
    angle wrapping or covariance validation.

    Validation requires a positive integer dimension, nonempty unique
    coordinate names, and unique in-bounds integer angular indices.
    """

    name: str
    coordinates: tuple[str, ...]
    dimension: int
    angle_indices: tuple[int, ...] = ()

    def __post_init__(self):
        dimension = _dimension(self.dimension)
        _identifier(self.name, "measurement-space name")
        coordinates = tuple(self.coordinates)
        if len(coordinates) != dimension:
            raise ValueError("coordinates must match measurement dimension")
        for coordinate in coordinates:
            _identifier(coordinate, "coordinate name")
        if len(set(coordinates)) != dimension:
            raise ValueError("coordinate names must be unique")
        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "angle_indices", _angles(self.angle_indices, dimension))


@dataclass(frozen=True)
class MeasurementMoments:
    """Observation moments without a state cross-covariance."""

    mean: np.ndarray
    covariance: np.ndarray

    def __post_init__(self):
        mean = _vector(self.mean, "predicted observation mean")
        covariance = _covariance(
            self.covariance,
            mean.size,
            "predicted observation covariance",
        )

        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "covariance", covariance)

@dataclass(frozen=True)
class GaussianState:
    """Store a detached state mean and covariance.

    Attributes
    ----------
    x:
        Estimated mean vector, mathematically m, with shape (n,).

    P:
        Estimated covariance matrix with shape (n, n).

    Copying
    -------
    Arrays are copied so modifying the caller's original arrays does
    not modify this container.

    Validation
    ----------
    Construction validates real finite arrays, compatible shapes,
    symmetry and positive semidefiniteness. validate_gaussian_state
    additionally checks an expected dimension and wraps angular means.

    The stored arrays remain writable despite frozen=True.

    This container carries no timestamp, coordinate-frame identifier
    or state-coordinate identity. Those must be established elsewhere.
    """

    x: np.ndarray
    P: np.ndarray

    def __post_init__(self):
        mean = _vector(self.x, "x")
        object.__setattr__(self, "x", mean)
        object.__setattr__(self, "P", _covariance(self.P, mean.size, "P"))


def validate_gaussian_state(x, P, state_dim, angle_indices=()):
    """Validate and copy a Gaussian representation.

    Checks:
        * The mean has the required vector shape.
        * The covariance has the corresponding square shape.
        * All entries are finite.
        * The covariance is approximately symmetric.
        * The covariance is positive semidefinite within tolerance.

    Small negative covariance eigenvalues attributed to numerical
    roundoff are clipped to zero. Substantially negative eigenvalues
    cause rejection.

    Angular mean components are wrapped into [-pi, pi).

    These checks establish numerical admissibility. They do not prove
    that the estimated distribution is calibrated or physically correct.

    Current assumptions:
        * state_dim is a valid positive dimension.
        * angle_indices contains valid, unique coordinate indices.
        * Inputs are intended to be real-valued.

    These metadata/input assumptions are explicitly validated.
    """
    state_dim = _dimension(state_dim, "state_dim")
    indices = _angles(angle_indices, state_dim)
    state = _vector(x, "x")
    if state.shape != (state_dim,):
        raise ValueError(f"x must have shape ({state_dim},)")
    covariance = _covariance(P, state_dim, "P")
    if indices:
        state[list(indices)] = (state[list(indices)] + np.pi) % (2 * np.pi) - np.pi
    return GaussianState(state, covariance)


@dataclass(frozen=True)
class MeasurementPrediction:
    """Store predicted observation moments before measurement updating.

    For the conventional filters, these correspond to:

        mean:
            Predicted observation mean, z_hat.

        covariance:
            Full predicted observation/innovation covariance, S.
            This generally includes propagated state uncertainty
            and measurement noise.

        cross_covariance:
            State-observation cross-covariance, C, shape (n_x, n_z).

        noise:
            Measurement-noise covariance, R, shape (n_z, n_z).
            This is only the noise contribution, not the full S.

        jacobian:
            Optional observation matrix or local Jacobian.
            A UKF can supply None because its moment propagation
            does not require an explicit observation Jacobian.

    This is a result container. It does not compute these quantities
    and validates/detaches its array fields. It does not check joint
    covariance consistency, which requires the state covariance too.
    """

    mean: np.ndarray
    covariance: np.ndarray
    cross_covariance: np.ndarray
    noise: np.ndarray
    jacobian: np.ndarray | None = None

    def __post_init__(self):
        readonly_fields = {field for field in self.__dataclass_fields__
                           if isinstance(getattr(self, field), np.ndarray)
                           and not getattr(self, field).flags.writeable}
        mean = _vector(self.mean, "predicted observation mean")
        cross = _real_array(self.cross_covariance, "cross covariance")
        if cross.ndim != 2 or cross.shape[0] < 1 or cross.shape[1] != mean.size:
            raise ValueError("cross covariance must have shape (state_dim, measurement_dim)")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "covariance", _covariance(self.covariance, mean.size, "innovation covariance"))
        object.__setattr__(self, "noise", _covariance(self.noise, mean.size, "measurement noise"))
        object.__setattr__(self, "cross_covariance", cross)
        if self.jacobian is not None:
            jacobian = _real_array(self.jacobian, "observation Jacobian")
            if jacobian.shape != (mean.size, cross.shape[0]):
                raise ValueError("Jacobian must have shape (measurement_dim, state_dim)")
            object.__setattr__(self, "jacobian", jacobian)
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            if field in readonly_fields and isinstance(value, np.ndarray):
                value.setflags(write=False)


@dataclass(frozen=True)
class InnovationDiagnostics:
    """Store compatibility diagnostics for one candidate observation.

    residual:
        Innovation y = z - z_hat, with appropriate angular handling
        performed by the code that constructs this result.

    covariance:
        Innovation covariance S.

    nis:
        Normalized innovation squared:

            NIS = y^T S^{-1} y.

    log_likelihood:
        Logarithm of the observation density evaluated at the recorded
        observation under the selected predictive model.

        For an IMM, covariance/NIS describe moment-matched gating geometry while
        log_likelihood describes the weighted component density. It need not equal
        a Gaussian density computed from the reported residual/covariance. Gaussian
        chi-square coverage is not exact for a general mixture.

        This is a log density, not a posterior mode probability.
        IMM combines per-mode likelihoods with prior mode probabilities.

    whitened_residual:
        For a Cholesky factorization S = L L^T, the vector L^{-1} y.
        Its squared Euclidean norm equals NIS.

    Association code can use these diagnostics for gating and costs
    without knowing the internal details of an EKF or UKF.

    Arrays are validated and copied. Algebraic agreement among NIS,
    whitening and likelihood is the producer's responsibility.
    """

    residual: np.ndarray
    covariance: np.ndarray
    nis: float
    log_likelihood: float
    whitened_residual: np.ndarray

    def __post_init__(self):
        readonly_fields = {field for field in self.__dataclass_fields__
                           if isinstance(getattr(self, field), np.ndarray)
                           and not getattr(self, field).flags.writeable}
        residual = _vector(self.residual, "innovation residual")
        whitened = _vector(self.whitened_residual, "whitened residual")
        if whitened.shape != residual.shape:
            raise ValueError("residual and whitened residual must have the same shape")
        nis = _real_array(self.nis, "NIS")
        if nis.ndim != 0 or nis < 0:
            raise ValueError("NIS must be a finite non-negative scalar")
        raw = np.asarray(self.log_likelihood)
        if np.iscomplexobj(raw) or raw.ndim != 0:
            raise ValueError("log likelihood must be a real scalar")
        log_likelihood = float(raw)
        # Negative infinity is allowed for an impossible/underflowed observation.
        if np.isnan(log_likelihood) or log_likelihood == np.inf:
            raise ValueError("log likelihood must not be NaN or positive infinity")
        object.__setattr__(self, "residual", residual)
        object.__setattr__(self, "whitened_residual", whitened)
        object.__setattr__(self, "covariance", _covariance(self.covariance, residual.size, "innovation covariance"))
        object.__setattr__(self, "nis", float(nis))
        object.__setattr__(self, "log_likelihood", log_likelihood)
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            if field in readonly_fields and isinstance(value, np.ndarray):
                value.setflags(write=False)


@dataclass(frozen=True)
class StateEstimate:
    """Expose an estimated mean/covariance with time and frame metadata.

    x:
        Estimated state mean.

    P:
        Estimated state covariance.

    timestamp_s:
        Time represented by the estimate, in seconds.
        None means no timestamp has been supplied.

    coordinate_frame:
        Identifier of the frame used to interpret the state.

    Unlike GaussianState, this type includes tracking metadata.

    Construction validates and detaches the arrays and checks timestamp
    and frame metadata. Stored arrays remain writable.

    For heterogeneous model states, coordinate_frame alone does not
    describe the complete state ordering or parameterization.
    """

    x: np.ndarray
    P: np.ndarray
    timestamp_s: float | None
    coordinate_frame: str

    def __post_init__(self):
        mean = _vector(self.x, "state estimate mean")
        object.__setattr__(self, "x", mean)
        object.__setattr__(self, "P", _covariance(self.P, mean.size, "state estimate covariance"))
        object.__setattr__(self, "timestamp_s", _timestamp(self.timestamp_s))
        _identifier(self.coordinate_frame, "state coordinate frame")


@dataclass(frozen=True)
class Measurement:
    """Carry an observation and its optional metadata into an estimator.

    value:
        Numerical observation vector z, shape (n_z,).

    covariance:
        Optional observation-noise covariance R.
        None means the consuming estimator must obtain its noise model
        elsewhere; it does not mean zero measurement uncertainty.

    timestamp_s:
        Optional capture time in seconds.

    coordinate_frame:
        Optional identifier for the observation's coordinate frame.
        Storing this identifier does not transform the observation.

    measurement_space_identity:
        Optional description of coordinate meanings, ordering and
        angular components.

    The value and supplied covariance are copied and made read-only
    to reduce accidental modification after construction.

    sensor_id:
        Optional sensor provenance, such as "front_radar". This does not
        select or apply a calibration automatically.

    Validation checks real finite arrays, covariance validity, identity
    dimension, timestamp and identifier strings. Angular values are not
    automatically wrapped; the sensor adapter defines that policy.

    Consumers remain responsible for enforcing their required identity,
    frame, timing and sensor-model conventions.
    """

    value: np.ndarray
    covariance: np.ndarray | None = None
    timestamp_s: float | None = None
    coordinate_frame: str | None = None
    measurement_space_identity: MeasurementSpaceIdentity | None = None

    # Provenance remains distinct from measurement coordinates and frame.
    # Appended to preserve existing positional constructor arguments.
    sensor_id: str | None = None

    def __post_init__(self):
        value = _vector(self.value, "measurement value")
        identity = self.measurement_space_identity
        if identity is not None:
            if not isinstance(identity, MeasurementSpaceIdentity):
                raise TypeError("measurement_space_identity must be a MeasurementSpaceIdentity")
            if value.size != identity.dimension:
                raise ValueError("measurement dimension must match its identity")
        value.setflags(write=False)
        object.__setattr__(self, "value", value)
        if self.covariance is not None:
            covariance = _covariance(self.covariance, value.size, "measurement covariance")
            covariance.setflags(write=False)
            object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "timestamp_s", _timestamp(self.timestamp_s))
        if self.coordinate_frame is not None:
            _identifier(self.coordinate_frame, "measurement coordinate frame")
        if self.sensor_id is not None:
            _identifier(self.sensor_id, "sensor_id")


# ---------------------------------------------------------------------------
# FUTURE THESIS EXTENSION: OPERATOR-BASED FILTERING
# ---------------------------------------------------------------------------
#
# GaussianState stores a belief through mean and covariance because these
# parameters completely determine a Gaussian distribution.
#
# A future operator-based filter may require a different belief container
# with the same architectural purpose: retaining the numerical distribution
# representation passed from one filtering step to the next.
#
# Its contents must follow the selected recursive filtering construction.
# Possibilities include:
#
# 1. Fixed-support probability representation
#
#       p_k(dx) ≈ sum_i alpha_i * delta_{s_i}(dx)
#
#    Store:
#       * support points s_i;
#       * nonnegative probability masses alpha_i summing to one;
#       * state-coordinate definitions and relevant time/frame metadata.
#
#    Here the weights are probability masses. If values instead represent
#    density samples, quadrature weights must also be accounted for.
#
# 2. Basis/feature representation
#
#    Store:
#       * coefficients describing the belief;
#       * the associated basis or a reference to its definition;
#       * any reference measure needed to interpret the representation.
#
#    Such coefficients need not individually be nonnegative probabilities.
#    Validity must be assessed through the represented distribution.
#
# The learned transition and observation models are distinct from the
# current belief. A belief container need not own the learned M/L models.
#
# NCP/eNCP would supply selected learned conditional models; our numerical
# filtering construction must still propagate, condition and normalize
# the belief.
#
# A sufficiently specified belief representation may support:
#       * expectations of integrable functions;
#       * means and covariances when the required moments exist;
#       * marginal/event probabilities;
#       * scalar marginal CDFs and quantiles.
#
# These are properties of the represented approximation. Their accuracy
# depends on learning, support/basis coverage and numerical inference.
#
# General multivariate states do not have a unique scalar-style quantile
# ordering. Quantiles should be defined for individual coordinates or
# specified scalar functions of the state.
#
# Do not choose the final learned-belief type until the recursive
# representation and its required operations have been specified.