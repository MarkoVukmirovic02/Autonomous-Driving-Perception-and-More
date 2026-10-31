"""Additive-noise UKF with separate state/observation sigma-point propagation.

The scaled Unscented Transform evaluates nonlinear functions at deterministic
integration nodes. Sigma points are not particles or independently maintained
motion hypotheses; signed weights need not be probabilities.

Prediction reconstructs a Gaussian and adds Q. Observation propagation then
regenerates points from that complete covariance, including process noise.
The update conditions an approximating joint Gaussian. Even exact first/second
moments would not make that conditioning exact for a general nonlinear law.

This implementation uses an eigen square root, allowing singular PSD covariance
without arbitrary jitter. It is not a square-root UKF: full covariances are
stored and the posterior covariance uses subtraction. Finite precision, signed
weights and broad uncertainty can still cause invalid covariances, which are
rejected if their negative eigenvalues exceed the shared numerical tolerance.
"""
import numpy as np
from .nonlinear_filter_base import (
    NonlinearFilterBase, MeasurementPrediction, covariance, matrix, vector,
    wrap, weighted_mean,
)
from Perception.tracking.types import validate_gaussian_state


def _parameter(value, name):
    """Require a finite real numerical scalar, excluding booleans/complex data."""
    raw = np.asarray(value)
    if raw.ndim != 0 or raw.dtype.kind not in 'iuf':
        raise ValueError(f'{name} must be a real numerical scalar')
    result = float(raw)
    if not np.isfinite(result):
        raise ValueError(f'{name} must be finite')
    return result


class UnscentedKalmanFilter(NonlinearFilterBase):
    """Gaussian filtering using a 2*n+1 scaled sigma-point integration rule.

    Model/callable and noise-provider configuration follows NonlinearFilterBase.
    Neither motion nor observation Jacobians are required. Models need not
    implement derivatives to use this filter.

    alpha>0 controls sigma-point spread; beta>=0 modifies the central covariance
    weight (beta=2 is conventional for Gaussian inputs); n+kappa must be positive.
    lambda=alpha**2*(n+kappa)-n and scale=n+lambda are computed at construction.
    Changing these public configuration fields later does not regenerate a
    consistent rule: construct a new configured filter to change its parameters.

    Very small alpha can create large signed weights and numerical cancellation.
    A parameter choice being accepted does not establish accuracy/stability for
    a given problem. Eigen-based node orientation can also affect nonlinear
    quadrature; it is not a claim of invariant exact nonlinear integration.

    State-dependent Q is evaluated once at the previous mean, as documented in
    the shared base. This is not augmented-noise propagation or integration of
    Q(X) across sigma points. Nonadditive noise requires a different construction.

    predict/update preserve detached (mean,covariance) tuple returns. Both
    validate candidates before committing belief/diagnostics; combined step
    inherits rollback of those fields, excluding external provider side effects.
    """
    def __init__(self, x0, P0, Q=None, R=None, f=None, h=None,
                 measurement_dim=None, alpha=0.3, beta=2.0, kappa=0.0, *,
                 motion_model=None, measurement_model=None,
                 state_angle_indices=None, measurement_angle_indices=None):
        alpha, beta, kappa = (_parameter(alpha, 'alpha'), _parameter(beta, 'beta'),
                              _parameter(kappa, 'kappa'))
        if alpha <= 0 or beta < 0:
            raise ValueError('alpha must be positive and beta non-negative')
        self._initialize(x0, P0, Q, R, f, h, motion_model, measurement_model,
                         measurement_dim, state_angle_indices, measurement_angle_indices)
        if self.n+kappa <= 0 or not np.isfinite(self.n+kappa):
            raise ValueError('n+kappa must be positive and finite')
        self.alpha, self.beta, self.kappa = alpha, beta, kappa
        try:
            self.scale = alpha**2*(self.n+kappa)
        except OverflowError as error:
            raise ValueError('Sigma-point scale exceeds numerical range') from error
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError('Sigma-point scale must be positive and finite')
        self.lambda_ = self.scale-self.n
        self.wm, self.wc = self._compute_weights()

    def _compute_weights(self):
        """Mean weights sum to one; covariance weights have a central correction.

        Wm[0]=lambda/scale, Wm[i]=1/(2*scale).
        Wc[0]=Wm[0]+1-alpha**2+beta, Wc[i]=Wm[i].
        Covariance weights are not required to sum to one.
        """
        try:
            with np.errstate(over='raise', invalid='raise', divide='raise'):
                wm = np.full(2*self.n+1, .5/self.scale)
                wc = wm.copy()
                wm[0] = self.lambda_/self.scale
                wc[0] = wm[0]+1-self.alpha**2+self.beta
        except (OverflowError, FloatingPointError) as error:
            raise ValueError('Unscented weights exceed numerical range') from error
        return vector(wm, 2*self.n+1, 'mean weights'), vector(wc, 2*self.n+1, 'covariance weights')

    def generate_sigma_points(self):
        """Return a detached (2*n+1,n) node array for the current Gaussian.

        For L@L.T=P, nodes are m and m +/- sqrt(scale)*L[:,i]. Here L is
        obtained from an eigendecomposition rather than Cholesky so PSD P is
        allowed. Before nonlinear transformation the rule reproduces the
        Euclidean mean/covariance, up to floating-point error.

        Angular wrapping uses a local chart. Broad angular uncertainty can
        invalidate Euclidean reconstruction across that chart.
        """
        current = self.state_snapshot()
        values, vectors = np.linalg.eigh(current.P)
        scale = _parameter(self.scale, 'sigma-point scale')
        if scale <= 0:
            raise ValueError('Sigma-point scale must be positive')
        try:
            with np.errstate(over='raise', invalid='raise'):
                root = vectors*(np.sqrt(scale)*np.sqrt(np.maximum(values, 0)))
                points = np.vstack([current.x, current.x+root.T, current.x-root.T])
        except FloatingPointError as error:
            raise FloatingPointError('Sigma-point construction exceeds numerical range') from error
        return wrap(points, self.state_angle_indices)

    def predict_sigma_points(self, sigma_points, dt):
        """Evaluate f(point,dt) for one complete native-state node set."""
        points = matrix(sigma_points, (2*self.n+1, self.n), 'sigma points')
        dt = self._dt(dt)
        return np.array([vector(self.f(point.copy(), dt), self.n, 'f(x)')
                         for point in points])

    def predict(self, dt):
        """Approximate state moments, reconstruct a Gaussian, then add Q.

        m_minus=sum Wm[i]*f(chi[i]).
        P_minus=sum Wc[i]*dx[i]@dx[i].T+Q.
        Local angular means/differences replace Euclidean ones where declared.
        The rule captures some curvature effects missed by first-order methods,
        but does not guarantee exact nonlinear moments or improved accuracy.
        """
        dt = self._dt(dt)
        Q = self._process_noise(dt)
        points = self.predict_sigma_points(self.generate_sigma_points(), dt)
        mean = weighted_mean(points, self.wm, self.state_angle_indices)
        weights = vector(self.wc, 2*self.n+1, 'covariance weights')
        with np.errstate(over='raise', invalid='raise'):
            errors = wrap(points-mean, self.state_angle_indices)
            P = covariance((errors.T*weights) @ errors+Q, self.n, 'predicted P')
        candidate = validate_gaussian_state(mean, P, self.n, self.state_angle_indices)
        self.x, self.P = candidate.x, candidate.P
        self.last_innovation = None
        return self._snapshot()

    def measurement_sigma_points(self, points):
        """Evaluate h at paired state nodes; preserve ordering for cross-covariance."""
        points = matrix(points, (2*self.n+1, self.n), 'sigma points')
        return np.array([vector(self.h(point.copy()), self.m, 'h(x)') for point in points])

    def predict_measurement(self, R=None):
        """Inspect approximate observation moments without changing the belief.

        Regenerate points from the CURRENT full Gaussian covariance. Reusing
        only the earlier motion-propagated points would omit the subsequently
        added Q from observation propagation in this formulation.

        z_hat=sum Wm*zeta; S=sum Wc*dz@dz.T+R; C=sum Wc*dx@dz.T.
        dx and dz must refer to the same point index. R is noise only; S is the
        full predicted observation/innovation covariance.
        """
        current = self.state_snapshot()
        noise = self._measurement_noise(R)
        points = self.generate_sigma_points()
        measurements = self.measurement_sigma_points(points)
        mean = weighted_mean(measurements, self.wm, self.measurement_angle_indices)
        weights = vector(self.wc, 2*self.n+1, 'covariance weights')
        dz = np.array([self._residual(z, mean) for z in measurements])
        with np.errstate(over='raise', invalid='raise'):
            dx = wrap(points-current.x, self.state_angle_indices)
            S = covariance((dz.T*weights) @ dz+noise, self.m, 'S')
            cross = (dx.T*weights) @ dz
        return MeasurementPrediction(mean, S, cross, noise)

    def update(self, z, R=None):
        """Condition the approximating joint Gaussian with a linear-system solve.

        K=C@S^-1, m_plus=m_minus+K@y,
        P_plus=P_minus-K@S@K.T.

        This is exact for a valid approximating joint Gaussian, not generally
        for the underlying nonlinear model. A Joseph form cannot be copied
        from EKF without specifying the corresponding effective linear/noise
        model. Materially invalid posterior covariance is rejected here.
        """
        z = vector(z, self.m, 'z')
        current = self.state_snapshot()
        prediction = self.predict_measurement(R=R)
        diagnostics = self._diagnostics(z, prediction)
        K = np.linalg.solve(prediction.covariance, prediction.cross_covariance.T).T
        with np.errstate(over='raise', invalid='raise'):
            mean = wrap(current.x+K @ diagnostics.residual, self.state_angle_indices)
            P = covariance(current.P-K @ prediction.covariance @ K.T, self.n, 'posterior P')
        candidate = validate_gaussian_state(mean, P, self.n, self.state_angle_indices)
        self.x, self.P = candidate.x, candidate.P
        self.last_innovation = diagnostics
        return self._snapshot()


# FUTURE OPERATOR-BASED EXTENSION
# -------------------------------
# UKF is a control for nonlinear moment integration without explicit Jacobians.
# Better performance than EKF alone may reflect quadrature rather than a richer
# distribution. Sigma points are not the learned belief support by default;
# reconstruction here still compresses the result into one Gaussian. A learned
# recursive construction must specify the distribution representation retained
# between steps and separate model improvements from numerical inference changes.
