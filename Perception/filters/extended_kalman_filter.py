"""Additive-noise Extended Kalman Filter with compatibility defaults.

The EKF maintains a single-Gaussian approximation to a nonlinear Bayesian
belief. It propagates the nominal mean through f and uses first derivatives
for local covariance propagation. Measurement updating conditions a locally
approximating joint Gaussian law, not the exact nonlinear joint law.

This module supplies those EKF-specific equations. NonlinearFilterBase owns
shared validation, covariance/provider policies, angular helpers and snapshots.
Prefer explicit motion/measurement models for new configurations. The default
four-state Cartesian motion and stationary-origin radar observation exist for
older callers; they are not generic sensor defaults for every state vector.
"""
import numpy as np
from .nonlinear_filter_base import (
    NonlinearFilterBase, MeasurementPrediction, covariance, matrix, vector, wrap,
)
from Perception.tracking.types import validate_gaussian_state
from Perception.motion_models.nonlinear.nonlinear_base import NonlinearMotionModel
from Perception.measurement_models.nonlinear.nonlinear_base import NonlinearMeasurementModel


class ExtendedKalmanFilter(NonlinearFilterBase):
    """Local first-order Gaussian prediction and measurement conditioning.

    Configuration:
        Models supply f/h and their Jacobians. If absent, supply callables
        f(x,dt), h(x), jacobian_f(x,dt), jacobian_h(x).
        Explicit Q/R or their supported providers override model covariances.
        See the shared base for provider timing and state-dependent Q policy.

    Compatibility defaults:
        Missing motion uses CV [px,py,vx,vy]. Missing observation uses planar
        stationary-origin [range,azimuth,positive-receding range_rate]. These
        defaults require four states; noise still must be supplied.
        dt, if supplied to construction, is retained as validated metadata;
        predict/step still require their explicit interval argument.

    predict/update return detached (mean,covariance) tuples, as before. They
    validate candidate results before committing the belief/innovation cache.
    The inherited combined step restores those fields on failure, excluding
    arbitrary side effects of external models/providers.

    Numerical success is not a guarantee of calibration, observability or
    adequate local linearization. Broad angular uncertainty and multiple modes
    are not fully described by this local Gaussian representation.
    """
    def __init__(self, x0, P0, Q=None, R=None, dt=None, *, f=None, h=None,
                 jacobian_f=None, jacobian_h=None, motion_model=None,
                 measurement_model=None, measurement_dim=None,
                 state_angle_indices=None, measurement_angle_indices=None):
        legacy_observation = measurement_model is None and h is None
        legacy_motion = motion_model is None and f is None
        if legacy_observation:
            h = self.h
            jacobian_h = self.jacobian_h if jacobian_h is None else jacobian_h
            measurement_dim = 3
            if measurement_angle_indices is None:
                measurement_angle_indices = (1,)
        if legacy_motion:
            f = self.f
            jacobian_f = self.jacobian_f if jacobian_f is None else jacobian_f
        self._initialize(x0, P0, Q, R, f, h, motion_model, measurement_model,
                         measurement_dim, state_angle_indices, measurement_angle_indices)
        if legacy_motion and self.n != 4:
            raise ValueError('Default CV dynamics require four states')
        if legacy_observation and self.n != 4:
            raise ValueError('Default radar measurement requires four states')
        if motion_model is not None:
            jacobian_f = ((lambda x, dt: motion_model.transition_matrix(dt))
                          if self._linear_motion else getattr(motion_model, 'jacobian', None))
        if measurement_model is not None:
            jacobian_h = getattr(measurement_model, 'jacobian', None)
        if not callable(jacobian_f) or not callable(jacobian_h):
            raise ValueError('EKF requires transition and measurement Jacobians')
        # An inherited placeholder is callable but is not an implementation.
        if getattr(jacobian_f, '__func__', None) is NonlinearMotionModel.jacobian:
            raise ValueError('Motion model must implement a transition Jacobian for EKF use')
        if getattr(jacobian_h, '__func__', None) is NonlinearMeasurementModel.jacobian:
            raise ValueError('Measurement model must implement an observation Jacobian for EKF use')
        self.jacobian_f, self.jacobian_h = jacobian_f, jacobian_h
        self.dt = None if dt is None else self._dt(dt)

    def f(self, x, dt):
        """Legacy Cartesian CV nominal motion; explicit models are preferred."""
        x, dt = vector(x, 4, 'default CV state'), self._dt(dt)
        return np.array([x[0]+dt*x[2], x[1]+dt*x[3], x[2], x[3]])

    def jacobian_f(self, x, dt):
        """Exact derivative of the legacy linear CV transition."""
        vector(x, 4, 'default CV state')
        dt = self._dt(dt)
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        return F

    def h(self, x):
        """Legacy stationary-origin projection using reviewed radar geometry.

        At range below 1e-6 m, bearing/rate are numerical zero placeholders;
        the returned range is the actual range, not an epsilon-clamped value.
        Prefer an explicit radar model with near_origin_policy='raise' when
        undefined geometry should be rejected. No sign/frame conversion occurs.
        """
        try:
            from Perception.measurement_models.nonlinear.radar import _planar_cartesian_projection
        except ModuleNotFoundError as error:
            if error.name != 'Perception.measurement_models.nonlinear.radar':
                raise
            raise ImportError(
                'Legacy radar defaults require the optional radar measurement module; supply an explicit measurement_model.'
            ) from error
        return _planar_cartesian_projection(vector(x, 4, 'default radar state'), 1e-6)

    def jacobian_h(self, x):
        """Derivative matching the legacy projection's near-origin policy."""
        try:
            from Perception.measurement_models.nonlinear.radar import _planar_cartesian_jacobian
        except ModuleNotFoundError as error:
            if error.name != 'Perception.measurement_models.nonlinear.radar':
                raise
            raise ImportError(
                'Legacy radar defaults require the optional radar measurement module; supply an explicit measurement_model.'
            ) from error
        return _planar_cartesian_jacobian(vector(x, 4, 'default radar state'), 1e-6)

    @staticmethod
    def normalize_angle(angle):
        """Compatibility scalar angular normalization into [-pi,pi)."""
        raw = np.asarray(angle)
        if raw.ndim != 0:
            raise ValueError('angle must be a scalar')
        return float(wrap(vector([angle], 1, 'angle'), (0,))[0])

    def predict(self, dt):
        """First-order approximate Chapman--Kolmogorov moment prediction.

        A=partial f/partial x evaluated at previous updated mean m_plus.
        m_minus=f(m_plus,dt).
        P_minus=A@P_plus@A.T+Q(m_plus,dt).

        Mean propagation omits curvature corrections, while Q is evaluated
        at the representative previous mean under the configured noise policy.
        This is generally not exact nonlinear moment integration.
        """
        dt = self._dt(dt)
        current = self.state_snapshot()
        A = matrix(self.jacobian_f(current.x.copy(), dt), (self.n, self.n), 'motion Jacobian')
        Q = self._process_noise(dt)
        mean = wrap(vector(self.f(current.x.copy(), dt), self.n, 'f(x)'), self.state_angle_indices)
        try:
            with np.errstate(over='raise', invalid='raise'):
                P = covariance(A @ current.P @ A.T+Q, self.n, 'predicted P')
        except FloatingPointError as error:
            raise FloatingPointError('EKF prediction exceeds numerical range') from error
        candidate = validate_gaussian_state(mean, P, self.n, self.state_angle_indices)
        self.x, self.P = candidate.x, candidate.P
        self.last_innovation = None
        return self._snapshot()

    def predict_measurement(self, R=None):
        """Inspect the locally linearized observation moments without updating.

        J=partial h/partial x at the CURRENT mean (normally m_minus).
        z_hat=h(m_minus), C=P_minus@J.T, S=J@P_minus@J.T+R.
        The observation Jacobian is evaluated after prediction, unlike A.
        h(m_minus) need not equal the exact predicted observation mean.
        """
        current = self.state_snapshot()
        noise = self._measurement_noise(R)
        mean = wrap(vector(self.h(current.x.copy()), self.m, 'h(x)'),
                    self.measurement_angle_indices)
        J = matrix(self.jacobian_h(current.x.copy()), (self.m, self.n), 'observation Jacobian')
        try:
            with np.errstate(over='raise', invalid='raise'):
                cross = current.P @ J.T
                S = covariance(J @ cross+noise, self.m, 'S')
        except FloatingPointError as error:
            raise FloatingPointError('EKF observation propagation exceeds numerical range') from error
        return MeasurementPrediction(mean, S, cross, noise, J)

    def update(self, z, R=None):
        """Gaussian conditioning under the locally affine observation model.

        y is the observation residual, with declared angles wrapped.
        K=C@S^-1 is computed by a linear solve, not an explicit inverse.
        m_plus=m_minus+K@y.
        Joseph covariance=(I-KJ)@P_minus@(I-KJ).T+K@R@K.T.

        The Joseph form supports numerical covariance stability; it cannot
        compensate for model error or inaccurate local linearization. The
        diagnostic log density is that of the Gaussian observation approximation.
        """
        z = vector(z, self.m, 'z')
        current = self.state_snapshot()
        prediction = self.predict_measurement(R=R)
        diagnostics = self._diagnostics(z, prediction)
        K = np.linalg.solve(prediction.covariance, prediction.cross_covariance.T).T
        try:
            with np.errstate(over='raise', invalid='raise'):
                mean = wrap(current.x+K @ diagnostics.residual, self.state_angle_indices)
                A = np.eye(self.n)-K @ prediction.jacobian
                P = covariance(A @ current.P @ A.T+K @ prediction.noise @ K.T,
                               self.n, 'posterior P')
        except FloatingPointError as error:
            raise FloatingPointError('EKF update exceeds numerical range') from error
        candidate = validate_gaussian_state(mean, P, self.n, self.state_angle_indices)
        self.x, self.P = candidate.x, candidate.P
        self.last_innovation = diagnostics
        return self._snapshot()


def radar_to_state(z):
    """Legacy single-return initialization in a stationary sensor-aligned frame.

    Convert position and retain only measured radial velocity. The unobserved
    tangential velocity is assigned ZERO as a working initialization, not inferred
    to be physically zero. This function does not propagate R into P or account
    for moving-sensor geometry. Use a dedicated initializer for those requirements.
    """
    r, theta, rate = vector(z, 3, 'radar measurement')
    if r < 0:
        raise ValueError('Radar range must be non-negative')
    return np.array([r*np.cos(theta), r*np.sin(theta),
                     rate*np.cos(theta), rate*np.sin(theta)])


class RadEKF:
    """Compatibility state-only wrapper around the legacy four-state EKF.

    Retains earlier predict()/update(z)/step(z) mean-only returns. It is not
    the general metadata-aware estimator interface. R uses range/bearing/
    positive-receding-rate coordinates; raw sensor conventions must be converted.

    Default P0/Q are historical heuristic choices, not calibrated uncertainty.
    P0 is not inferred from the initialization measurement covariance. Fixed Q
    does not automatically scale when dt changes. New configurations should use
    explicit models and an initializer with justified uncertainty assumptions.
    """
    def __init__(self, measurement, R, dt=1/30, *, Q=None, P0=None):
        self.dt = ExtendedKalmanFilter._dt(dt)
        self.ekf = ExtendedKalmanFilter(
            radar_to_state(measurement),
            np.diag([1., 1., 100., 100.]) if P0 is None else P0,
            np.diag([.1, .1, 1., 1.]) if Q is None else Q, R, self.dt)

    def predict(self):
        """Advance by configured dt and return a detached estimated mean."""
        return self.ekf.predict(self.dt)[0]

    def update(self, measurement):
        """Update at the current time; this wrapper does not schedule timestamps."""
        return self.ekf.update(measurement)[0]

    def step(self, measurement):
        """Combined step using the shared belief/diagnostic rollback guarantee."""
        return self.ekf.step(measurement, self.dt)[0]


# FUTURE OPERATOR-BASED EXTENSION
# -------------------------------
# The EKF is a benchmark for local first-order inference with analytical models.
# Learned conditional laws can change model quality, while retaining this
# Gaussian update may discard richer distributional structure. Compare model
# changes separately from changes in belief representation/numerical inference.
# Learned transition/likelihood matrices on distribution coefficients are not
# direct replacements for this local gain or its state-coordinate Jacobians.
