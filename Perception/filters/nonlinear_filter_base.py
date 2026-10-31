"""Shared additive-noise Gaussian-filter infrastructure for EKF and UKF.

This module supplies validation, model/callable configuration, angular helpers,
noise selection, diagnostics and snapshots. Subclasses implement prediction and
update mathematics. It is not a complete estimator on its own.

Both filters approximate a generally nonlinear/non-Gaussian belief by a mean
and covariance. In code x is the estimated mean; P is its modeled covariance.
A Gaussian innovation log density is an approximation unless the corresponding
joint observation/state model is genuinely Gaussian.

The supported equations are x_next=f(x,dt)+w and z=h(x)+v. Nonadditive driving
noise requires extra Jacobians or augmented sigma points; this base does not
implement that extension. Known context must be configured explicitly in the
models/callables and aligned with capture time by the coordinating caller.
"""
import numpy as np
from Perception.tracking.types import (
    InnovationDiagnostics, Measurement, MeasurementPrediction,
    validate_gaussian_state, _angles, _covariance, _dimension, _real_array,
)


def vector(value, size, name):
    """Detach a finite real vector of exact shape; never flatten input matrices."""
    size = _dimension(size, f'{name} dimension')
    result = _real_array(value, name)
    if result.shape != (size,):
        raise ValueError(f'{name} must be a finite vector of shape ({size},)')
    return result


def matrix(value, shape, name):
    """Detach a finite real matrix with the requested rectangular/square shape."""
    shape = tuple(shape)
    if len(shape) != 2:
        raise ValueError('Matrix shape must have two dimensions')
    shape = tuple(_dimension(i, 'matrix dimension') for i in shape)
    result = _real_array(value, name)
    if result.shape != shape:
        raise ValueError(f'{name} must be a finite matrix of shape {shape}')
    return result


def covariance(value, size, name):
    """Use the shared symmetry/PSD policy; singular covariances are valid.

    Tiny numerical negative eigenvalues are clipped, not material model errors.
    Innovation solves separately require a positive-definite covariance.
    """
    return _covariance(value, _dimension(size, 'covariance dimension'), name)


def wrap(value, indices):
    """Wrap selected final-axis coordinates to [-pi,pi), returning a copy.

    Accept a vector or a stack of vectors, as required for sigma points. This
    selects angular representatives; it is not a circular probability model.
    """
    result = _real_array(value, 'angular values')
    if result.ndim == 0 or result.shape[-1] < 1:
        raise ValueError('Expected a vector or stack with a nonempty final axis')
    indices = _angles(indices, result.shape[-1])
    if indices:
        selected = list(indices)
        result[..., selected] = (result[..., selected]+np.pi) % (2*np.pi)-np.pi
    return result


def weighted_mean(points, weights, angles):
    """Reconstruct a mean using numerical weights and a local angular chart.

    points has shape (point_count, coordinate_count). Signed unscented weights
    are permitted: they are integration weights, not probabilities of points.
    Their sum must be one within a cancellation-scaled numerical tolerance.

    Angular offsets are taken relative to the first point before weighting.
    This avoids cancellation in circular sine/cosine means with signed weights,
    but remains a LOCAL approximation. Broad/wrapped angular uncertainty is not
    resolved by this convention; its chart dependence must be evaluated.
    """
    points = _real_array(points, 'points')
    if points.ndim != 2 or min(points.shape) < 1:
        raise ValueError('points must be a nonempty two-dimensional array')
    weights = vector(weights, points.shape[0], 'mean weights')
    angles = _angles(angles, points.shape[1])
    tolerance = 1e-10*max(1., float(np.sum(np.abs(weights))))
    if not np.isclose(weights.sum(), 1., rtol=0, atol=tolerance):
        raise ValueError('Mean weights must sum to one')
    with np.errstate(over='raise', invalid='raise'):
        mean = weights @ points
        for index in angles:
            offsets = (points[:, index]-points[0, index]+np.pi) % (2*np.pi)-np.pi
            mean[index] = points[0, index]+weights @ offsets
    return wrap(mean, angles)


class NonlinearFilterBase:
    """Common services; subclasses supply predict/update/predict_measurement.

    Models take precedence over f/h callables when both are supplied, preserving
    the existing interface. Explicit Q/R take precedence over model covariances.
    This precedence is configuration policy, not a combination of noise sources.

    Covariance providers:
        Q: fixed matrix, callable Q(mean_before_prediction,dt), or model noise.
        R: fixed matrix, no-argument callable R(), or model noise.
        Per-update R overrides all configured observation noise for that call.

    Q evaluated at a mean is an approximation when the disturbance covariance
    depends on uncertain state. Exact moment prediction would include the
    appropriate expectation of conditional noise covariance. Calling Q once at
    the mean does not perform that integral, even in sigma-point propagation.

    Shared numeric/model providers are expected not to mutate external context
    during evaluation. Copies protect x/P storage but cannot prevent arbitrary
    side effects in custom callables or make this mutable object thread-safe.
    """
    @property
    def measurement_space_identity(self):
        """Forward declared semantics; callable-only filters do not invent them."""
        if getattr(self.measurement_model, 'measurement_space_identity', None) is None:
            return None
        validator = getattr(self.measurement_model, 'validate_measurement_space_identity', None)
        return validator() if callable(validator) else None

    def _initialize(self, x0, P0, Q, R, f, h, motion_model,
                    measurement_model, measurement_dim, state_angle_indices,
                    measurement_angle_indices):
        """Configure the common model and validated initial belief.

        Observation dimension can be inferred from one h(initial_mean) call.
        Thus a callable used for inference must be valid at initialization and
        have no unwanted side effects. f is evaluated later with a supplied dt.
        """
        initial = _real_array(x0, 'x0')
        if initial.ndim != 1 or initial.size == 0:
            raise ValueError('x0 must be a nonempty one-dimensional vector')
        self.n = initial.size
        self.motion_model, self.measurement_model = motion_model, measurement_model
        state_angles = (state_angle_indices if state_angle_indices is not None
                        else getattr(motion_model, 'angle_indices', ()))
        measurement_angles = (measurement_angle_indices if measurement_angle_indices is not None
                              else getattr(measurement_model, 'angle_indices', ()))
        self._linear_motion = False
        if motion_model is not None:
            if _dimension(motion_model.state_dim, 'motion state_dim') != self.n:
                raise ValueError('motion model state dimension mismatch')
            self._linear_motion = callable(getattr(motion_model, 'transition_matrix', None))
            if self._linear_motion:
                # Validate third-party F before applying it to a state vector.
                f = lambda x, dt: matrix(motion_model.transition_matrix(dt),
                                         (self.n, self.n), 'F') @ x
            else:
                f = getattr(motion_model, 'predict_state', None)
        if measurement_model is not None:
            if _dimension(measurement_model.state_dim, 'measurement state_dim') != self.n:
                raise ValueError('measurement model state dimension mismatch')
            h = getattr(measurement_model, 'predict_measurement', None)
            measurement_dim = measurement_model.measurement_dim
        if not callable(f) or not callable(h):
            raise ValueError('Provide motion/measurement models or f and h callables')
        self.f, self.h = f, h
        self.state_angle_indices = _angles(state_angles, self.n)
        self.x = wrap(initial, self.state_angle_indices)
        self.P = covariance(P0, self.n, 'P0')
        if measurement_dim is None:
            example = _real_array(h(self.x.copy()), 'h(initial state)')
            if example.ndim != 1 or example.size == 0:
                raise ValueError('h must return a nonempty one-dimensional observation')
            measurement_dim = example.size
        self.m = _dimension(measurement_dim, 'measurement_dim')
        self.measurement_angle_indices = _angles(measurement_angles, self.m)
        self.Q = Q if callable(Q) or Q is None else covariance(Q, self.n, 'Q')
        self.R = R if callable(R) or R is None else covariance(R, self.m, 'R')
        if Q is None and motion_model is None:
            raise ValueError('Provide Q or a motion model')
        if R is None and measurement_model is None:
            raise ValueError('Provide R or a measurement model')
        if self.measurement_space_identity is not None:
            identity = self.measurement_space_identity
            if identity.dimension != self.m or identity.angle_indices != self.measurement_angle_indices:
                raise ValueError('Observation identity disagrees with filter metadata')
        self.last_innovation = None

    @staticmethod
    def _dt(dt):
        """Validate an explicit finite nonnegative numerical scalar interval."""
        raw = np.asarray(dt)
        if raw.ndim != 0 or raw.dtype.kind not in 'iuf':
            raise ValueError('dt must be a real numerical scalar')
        interval = float(raw)
        if not np.isfinite(interval) or interval < 0:
            raise ValueError('dt must be finite and non-negative')
        return interval

    def _process_noise(self, dt):
        """Evaluate configured Q at the previous mean, before state propagation."""
        dt = self._dt(dt)
        if self.Q is not None:
            value = self.Q(self.x.copy(), dt) if callable(self.Q) else self.Q
        elif self._linear_motion:
            value = self.motion_model.process_noise(dt)
        else:
            value = self.motion_model.process_noise(self.x.copy(), dt)
        return covariance(value, self.n, 'Q')

    def _measurement_noise(self, R):
        """Select a detached observation covariance without changing configuration."""
        value = R
        if value is None:
            value = self.R() if callable(self.R) else self.R
        if value is None:
            value = self.measurement_model.measurement_noise()
        return covariance(value, self.m, 'R')

    def _residual(self, z, mean):
        """Compute model residual or Euclidean subtraction, then wrap angles."""
        z, mean = vector(z, self.m, 'z'), vector(mean, self.m, 'predicted observation')
        if self.measurement_model is not None:
            residual = self.measurement_model.residual(z.copy(), mean.copy())
        else:
            with np.errstate(over='raise', invalid='raise'):
                residual = z-mean
        return wrap(vector(residual, self.m, 'residual'), self.measurement_angle_indices)

    def innovation(self, z, R=None):
        """Inspect Gaussian innovation diagnostics at the current belief, without updating."""
        return self._diagnostics(z, self.predict_measurement(R=R))

    def innovation_diagnostics(self, z, R=None):
        """Common non-mutating diagnostics interface."""
        return self.innovation(z, R=R)

    def measurement_prediction(self, R=None):
        """Common alias for the subclass's predicted observation moments."""
        return self.predict_measurement(R=R)

    def _diagnostics(self, z, prediction):
        """Cholesky-whiten residuals and evaluate a Gaussian predictive density.

        NIS=y.T@S^-1@y. Log density includes both NIS and logdet(S); it is not
        a posterior probability. These values inherit the filter's Gaussian
        moment approximation and local angular conventions.
        """
        if not isinstance(prediction, MeasurementPrediction):
            raise TypeError('Expected MeasurementPrediction')
        residual = self._residual(z, prediction.mean)
        S = covariance(prediction.covariance, self.m, 'innovation covariance')
        try:
            factor = np.linalg.cholesky(S)
        except np.linalg.LinAlgError as error:
            raise ValueError('Innovation covariance must be positive definite') from error
        whitened = np.linalg.solve(factor, residual)
        with np.errstate(over='raise', invalid='raise'):
            nis = float(whitened @ whitened)
            logdet = 2*np.log(np.diag(factor)).sum()
            likelihood = -.5*(self.m*np.log(2*np.pi)+logdet+nis)
        return InnovationDiagnostics(residual, S, nis, float(likelihood), whitened)

    def _snapshot(self):
        """Legacy (mean,covariance) return format, detached from filter storage."""
        return self.x.copy(), self.P.copy()

    def state_snapshot(self):
        """Validated native Gaussian summary; timing/frame context lives outside."""
        return validate_gaussian_state(self.x, self.P, self.n, self.state_angle_indices)

    def replace_state(self, x, P):
        """Install a validated belief and invalidate the previous innovation."""
        replacement = validate_gaussian_state(x, P, self.n, self.state_angle_indices)
        self.x, self.P = replacement.x.copy(), replacement.P.copy()
        self.last_innovation = None
        return self.state_snapshot()

    def step(self, z, dt, R=None):
        """Predict/update with rollback of this base's belief and diagnostic state.

        Validate direct inputs first. On subclass failure restore x, P and
        last_innovation. External provider side effects and any future extra
        subclass caches are outside this guarantee and need an extended contract.
        """
        z, dt = vector(z, self.m, 'z'), self._dt(dt)
        # Explicit overrides can be checked now. Defer configured providers
        # until update, preserving their evaluation at the predicted belief.
        noise = None if R is None else covariance(R, self.m, 'R')
        before = self.state_snapshot()
        diagnostic = self.last_innovation
        try:
            self.predict(dt)
            return self.update(z, R=noise)
        except Exception:
            self.x, self.P = before.x, before.P
            self.last_innovation = diagnostic
            raise

    def update_measurement(self, measurement):
        """Require matching observation semantics before the ordinary update.

        This numerical boundary does not enforce capture time/frame/source;
        those checks belong to the coordinating tracker and input adapter.
        """
        if not isinstance(measurement, Measurement):
            raise TypeError('measurement must be a Measurement')
        identity = measurement.measurement_space_identity
        if identity is None:
            raise ValueError('managed measurement must declare a MeasurementSpaceIdentity')
        expected = self.measurement_space_identity
        if expected is None or identity != expected:
            raise ValueError('measurement space does not match estimator')
        vector(measurement.value, self.m, 'measurement')
        return self.update(measurement.value, R=measurement.covariance)

    def predicted_measurement(self, R=None):
        """Legacy tuple alias: mean, innovation covariance, cross-covariance."""
        result = self.predict_measurement(R=R)
        return result.mean, result.covariance, result.cross_covariance


# FUTURE OPERATOR-BASED EXTENSION
# -------------------------------
# Shared timing-independent checks and coordinate conventions remain useful.
# A learned conditional law need not fit the additive/Gaussian representation
# used here. Introduce a belief/operator interface matched to the recursive
# construction rather than forcing every distribution into x/P. Learned model
# quality, moment approximation and belief representation are separate choices.
