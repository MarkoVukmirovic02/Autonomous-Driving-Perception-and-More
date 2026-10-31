"""Conventional linear Kalman filtering with validated Gaussian snapshots.

For the model x_k=F_k*x_(k-1)+w_k, z_k=H_k*x_k+v_k, a Gaussian initial
belief and mutually independent zero-mean Gaussian disturbances independent
of that initial state yield exact Gaussian Bayesian filtering. Matrices and
covariances are treated as known. This implementation has no explicit B*u
or affine observation offset; those require a deliberate model extension.

With suitable zero cross-covariances and finite second moments, the same
linear recursions have a linear minimum-mean-square-error interpretation
without Gaussianity. They then need not equal the unrestricted posterior
mean, and x/P do not describe the full posterior distribution.

In code, x denotes the estimated mean (m in the thesis), not ground truth.
P denotes its modeled error covariance. The filter handles numerical inference;
adapters/trackers handle raw data conversion, time, frame and sensor context.
"""
import numpy as np

from Perception.motion_models.linear.linear_base import LinearMotionModel
from Perception.measurement_models.linear.linear_base import LinearMeasurementModel
from Perception.tracking.types import (
    InnovationDiagnostics, Measurement, MeasurementPrediction,
    validate_gaussian_state, _covariance, _dimension, _real_array, _vector,
)


class KalmanFilter:
    """Generic linear estimator with Joseph-form covariance updates.

    Constructor arguments are detached; all model outputs are checked at the
    numerical boundary. x0 must have shape (state_dim,), not a row/column matrix.
    State and observation dimensions/order must agree with the configured models.

    Public x/P remain writable for compatibility. Prefer replace_state to make
    deliberate changes because it validates and clears stale diagnostics.
    Returned states/results are detached from filter storage.

    predict/update compute and validate candidates before committing x/P and
    diagnostic caches. step additionally restores this filter's state/caches
    if its combined prediction/update fails. This does not roll back external
    side effects in custom model implementations or provide thread safety.

    Exact Gaussian claims concern Euclidean linear models. Angular metadata
    supports local wrapping conventions, not exact global Gaussian inference
    on a circle or a general nonlinear/periodic observation law.
    """
    _CACHE_NAMES = ('innovation', 'innovation_covariance', 'kalman_gain',
                    'predicted_measurement', 'last_innovation')

    def __init__(self, x0: np.ndarray, P0: np.ndarray,
                 motion_model: LinearMotionModel,
                 measurement_model: LinearMeasurementModel) -> None:
        if not isinstance(motion_model, LinearMotionModel):
            raise TypeError('KalmanFilter requires a LinearMotionModel.')
        if not isinstance(measurement_model, LinearMeasurementModel):
            raise TypeError('KalmanFilter requires a LinearMeasurementModel.')
        self.motion_model = motion_model
        self.measurement_model = measurement_model
        self.state_dim = _dimension(motion_model.state_dim, 'state_dim')
        if measurement_model.state_dim != self.state_dim:
            raise ValueError('Motion and measurement state dimensions must agree.')
        self.state_angle_indices = tuple(getattr(motion_model, 'angle_indices', ()))
        self.measurement_angle_indices = tuple(getattr(measurement_model, 'angle_indices', ()))
        initial = validate_gaussian_state(x0, P0, self.state_dim, self.state_angle_indices)
        self.x, self.P = initial.x.copy(), initial.P.copy()
        self.I = np.eye(self.state_dim)
        self._clear_diagnostics()
        # Exercise H/R validation before exposing an incorrectly configured filter.
        self._validate_models()

    def _clear_diagnostics(self):
        for name in self._CACHE_NAMES:
            setattr(self, name, None)

    @property
    def measurement_space_identity(self):
        """Forward declared observation semantics; legacy unidentified models return None."""
        if getattr(self.measurement_model, 'measurement_space_identity', None) is None:
            return None
        return self.measurement_model.validate_measurement_space_identity()

    def _validate_initial_state(self):
        """Compatibility helper: validate/install the current numerical state."""
        state = validate_gaussian_state(self.x, self.P, self.state_dim,
                                         self.state_angle_indices)
        self.x, self.P = state.x, state.P

    def _validate_models(self):
        if (_dimension(self.motion_model.state_dim, 'motion state_dim') != self.state_dim
                or _dimension(self.measurement_model.state_dim, 'measurement state_dim')
                != self.state_dim):
            raise ValueError('Model state dimensions must match the filter.')
        self._observation_matrix(self.x)
        self._measurement_noise()
        # Validate angle metadata even when semantic identity is unspecified.
        if self.measurement_model._measurement_angles() != self.measurement_angle_indices:
            raise ValueError('Measurement angular metadata is inconsistent.')
        if getattr(self.measurement_model, 'measurement_space_identity', None) is not None:
            self.measurement_space_identity

    def _observation_matrix(self, x):
        # For this linear contract the Jacobian is exactly H. Obtain one H for
        # each operation; do not evaluate h independently with a second matrix.
        H = _real_array(self.measurement_model.jacobian(x), 'H')
        nz = _dimension(self.measurement_model.measurement_dim, 'measurement_dim')
        if H.shape != (nz, self.state_dim):
            raise ValueError(f'H must have shape ({nz}, {self.state_dim}).')
        return H

    def _measurement_noise(self, R=None):
        """Use per-observation R when supplied, otherwise the configured model R.

        The override does not reconfigure the model. PSD R is allowed even
        though innovation diagnostics/update require positive-definite S.
        """
        value = self.measurement_model.measurement_noise() if R is None else R
        nz = _dimension(self.measurement_model.measurement_dim, 'measurement_dim')
        return _covariance(value, nz, 'R')

    def state_snapshot(self):
        """Return validated detached native-state mean/covariance."""
        return validate_gaussian_state(self.x, self.P, self.state_dim,
                                       self.state_angle_indices)

    def replace_state(self, x, P):
        """Install a validated belief at the existing time; clear old update caches.

        Useful for IMM interaction. This operation itself has no timestamp;
        the coordinating tracker owns that metadata.
        """
        state = validate_gaussian_state(x, P, self.state_dim, self.state_angle_indices)
        self.x, self.P = state.x.copy(), state.P.copy()
        self._clear_diagnostics()
        return self.state_snapshot()

    def predict(self, dt: float) -> np.ndarray:
        """Evaluate the linear Chapman--Kolmogorov moment prediction.

        m_minus = F@m_plus
        P_minus = F@P_plus@F.T+Q

        Q is the state disturbance covariance accumulated over this interval.
        Its time dependence follows the concrete disturbance model, not a
        universal rule. No observation is incorporated and no noise is sampled.
        """
        dt = self.motion_model.validate_dt(dt)
        current = self.state_snapshot()
        F = _real_array(self.motion_model.transition_matrix(dt), 'F')
        if F.shape != (self.state_dim, self.state_dim):
            raise ValueError('F must match the square state dimension.')
        Q = _covariance(self.motion_model.process_noise(dt), self.state_dim, 'Q')
        try:
            with np.errstate(over='raise', invalid='raise'):
                mean = F @ current.x
                covariance = F @ current.P @ F.T+Q
        except FloatingPointError as error:
            raise FloatingPointError('Kalman prediction exceeds numerical range.') from error
        candidate = validate_gaussian_state(mean, covariance, self.state_dim,
                                             self.state_angle_indices)
        self.x, self.P = candidate.x, candidate.P
        self._clear_diagnostics()
        return self.x.copy()

    def measurement_prediction(self, R=None) -> MeasurementPrediction:
        """Inspect predicted observation moments without changing the belief.

        z_hat=H@m, C=P@H.T, S=H@P@H.T+R.
        S includes both predicted-state and observation-noise uncertainty.
        """
        current = self.state_snapshot()
        H = self._observation_matrix(current.x)
        noise = self._measurement_noise(R)
        try:
            with np.errstate(over='raise', invalid='raise'):
                mean = self.measurement_model.validate_measurement(H @ current.x)
                cross = current.P @ H.T
                S = H @ cross+noise
        except FloatingPointError as error:
            raise FloatingPointError('Observation moment prediction exceeds numerical range.') from error
        return MeasurementPrediction(mean, .5*(S+S.T), cross, noise, H)

    @staticmethod
    def innovation_diagnostics_from_values(residual, covariance):
        """Compute NIS and Gaussian predictive log density using Cholesky solves.

        For S=L@L.T, whitened=L^-1@y and NIS=whitened.T@whitened.
        log p(z|past)=-0.5*(n_z*log(2*pi)+logdet(S)+NIS).
        The value is a log density, not a posterior mode probability or a
        direct measure of sensor reliability. Correlated coordinates are allowed.
        """
        residual = _vector(residual, 'innovation residual')
        S = _covariance(covariance, residual.size, 'innovation covariance')
        try:
            factor = np.linalg.cholesky(S)
        except np.linalg.LinAlgError as error:
            raise ValueError('Innovation covariance must be positive definite') from error
        whitened = np.linalg.solve(factor, residual)
        with np.errstate(over='raise', invalid='raise'):
            nis = float(whitened @ whitened)
            logdet = 2*np.log(np.diag(factor)).sum()
            likelihood = -.5*(residual.size*np.log(2*np.pi)+logdet+nis)
        return InnovationDiagnostics(residual, S, nis, float(likelihood), whitened)

    def innovation_diagnostics(self, z, R=None) -> InnovationDiagnostics:
        """Non-mutating compatibility query used by gating and mode comparison."""
        measurement = self.measurement_model.validate_measurement(z)
        prediction = self.measurement_prediction(R)
        residual = self.measurement_model.residual(measurement, prediction.mean)
        return self.innovation_diagnostics_from_values(residual, prediction.covariance)

    def update(self, z: np.ndarray, R=None) -> np.ndarray:
        """Condition on a measurement already in the declared observation space.

        K=C@S^-1, m_plus=m_minus+K@y.
        Joseph covariance: P_plus=(I-KH)@P_minus@(I-KH).T+K@R@K.T.

        Solve a linear system rather than explicitly forming S^-1. The
        Joseph expression helps numerical PSD preservation, but does not fix
        incorrect physical models, measurement origins or miscalibrated covariances.
        All validation/diagnostics finish before committing the updated belief.
        """
        measurement = self.measurement_model.validate_measurement(z)
        current = self.state_snapshot()
        prediction = self.measurement_prediction(R)
        H, noise, S = prediction.jacobian, prediction.noise, prediction.covariance
        residual = self.measurement_model.residual(measurement, prediction.mean)
        diagnostic = self.innovation_diagnostics_from_values(residual, S)
        # Positive-definiteness has already been checked by the diagnostics.
        K = np.linalg.solve(S, prediction.cross_covariance.T).T
        try:
            with np.errstate(over='raise', invalid='raise'):
                mean = current.x+K @ residual
                I_KH = self.I-K @ H
                covariance = I_KH @ current.P @ I_KH.T+K @ noise @ K.T
        except FloatingPointError as error:
            raise FloatingPointError('Kalman update exceeds numerical range.') from error
        candidate = validate_gaussian_state(mean, covariance, self.state_dim,
                                             self.state_angle_indices)
        # Preserve legacy public diagnostics used by existing consumers.
        self.x, self.P = candidate.x, candidate.P
        self.predicted_measurement = prediction.mean.copy()
        self.innovation = residual.copy()
        self.innovation_covariance = S.copy()
        self.kalman_gain = K.copy()
        self.last_innovation = diagnostic
        return self.x.copy()

    def update_measurement(self, measurement):
        """Managed update with required semantic identity.

        Time/frame/source selection remains in adapters/trackers. A numeric
        filter cannot establish those relationships from values alone.
        """
        if not isinstance(measurement, Measurement):
            raise TypeError('measurement must be a Measurement')
        identity = measurement.measurement_space_identity
        if identity is None:
            raise ValueError('managed measurement must declare a MeasurementSpaceIdentity')
        if self.measurement_space_identity is None or identity != self.measurement_space_identity:
            raise ValueError('measurement space does not match estimator')
        self.measurement_model.validate_measurement(measurement.value)
        return self.update(measurement.value, R=measurement.covariance)

    def step(self, z: np.ndarray, dt: float, R=None) -> np.ndarray:
        """Predict then update, restoring this filter's belief/caches on failure.

        Direct predict/update remain separately available. This helper does
        not schedule timestamps, process missing detections or prevent repeated
        incorporation of a measurement; those are tracker responsibilities.
        """
        measurement = self.measurement_model.validate_measurement(z)
        dt = self.motion_model.validate_dt(dt)
        noise = self._measurement_noise(R)
        before = self.state_snapshot()
        caches = {name: getattr(self, name) for name in self._CACHE_NAMES}
        try:
            self.predict(dt)
            return self.update(measurement, R=noise)
        except Exception:
            self.x, self.P = before.x, before.P
            for name, value in caches.items():
                setattr(self, name, value)
            raise


# FUTURE OPERATOR-BASED EXTENSION
# -------------------------------
# This filter is an exact linear-Gaussian control case for a learned recursive
# construction. It does not learn the transition or observation laws.
# Learned matrices on belief weights/features are not substitutes for the
# physical F/H matrices or direct equivalents of K. Their full prediction and
# normalized likelihood reweighting implement analogous Bayesian operations
# in a different distribution representation. Compare filtering/calibration,
# conditional-model accuracy and computational approximation separately.
