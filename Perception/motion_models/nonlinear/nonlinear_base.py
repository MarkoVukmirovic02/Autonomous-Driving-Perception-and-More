"""Validated interface for conventional nonlinear motion models.

The nominal transition f(x, dt) advances one specified physical state.
A filter separately propagates a probability distribution through that model.
In general E[f(X)] != f(E[X]); predict_state is therefore not an exact
nonlinear mean-prediction operation.

The associated conditional disturbance model may be written as

    X_next = f(X_previous, dt) + W,
    E[W | X_previous=x, past information] = 0,
    Cov(W | X_previous=x, past information) = Q(x, dt).

If Gaussian conditional disturbances are assumed, this specifies a Gaussian
transition law at each fixed previous state (possibly singular). It does not
make the predicted belief Gaussian. When Q depends on x, W generally is not
independent of the previous state; a construction W=G(x,dt)*epsilon may instead
use an independent driving variable epsilon.

This interface supplies f, Q and optionally a Jacobian. It does not sample
noise, enforce temporal independence, manage observations or store a belief.
Known controls/context are not explicit arguments and require a separately
specified extension when needed.
"""
from abc import ABC, abstractmethod

import numpy as np


def _real_array(value, name):
    """Reject lossy/non-numerical coercions before conversion to floating point."""
    raw = np.asarray(value)
    if raw.dtype.kind not in 'iuf':
        raise ValueError(f"{name} must contain real numerical values.")
    result = np.asarray(raw, dtype=float)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite.")
    return result.copy()


class NonlinearMotionModel(ABC):
    """Input/output validation around nonlinear model implementations.

    Concrete models define name, positive integer state_dim, state ordering,
    coordinate units and disturbance assumptions. Type annotations alone do
    not create those attributes. Optional state_angle_indices metadata is
    interpreted by filters; this base does not wrap arbitrary state entries.

    Public predict_state/process_noise validate inputs and outputs. Subclasses
    implement _predict_state/_process_noise using validated copies, keeping
    validation separate from the physical formulas. Call the public methods
    from filters rather than bypassing them through the underscore methods.

    A subclass providing jacobian must validate its inputs and call
    validate_jacobian on its output: the base cannot intercept an override.
    Jacobians are optional so derivative-free filters can use models without
    them. EKF use requires a concrete, correct implementation.
    """
    name: str
    state_dim: int

    def _state_dimension(self):
        dimension = self.state_dim
        if (isinstance(dimension, (bool, np.bool_))
                or not isinstance(dimension, (int, np.integer))):
            raise TypeError("state_dim must be an integer.")
        if dimension < 1:
            raise ValueError("state_dim must be positive.")
        return int(dimension)

    @staticmethod
    def normalize_angle(angle: float) -> float:
        """Wrap a finite scalar angle in radians to [-pi, pi).

        +pi maps to -pi. This chooses a coordinate representative; it does not
        represent broad circular uncertainty or remove the branch cut used
        when differentiating an angular transition.
        """
        raw = np.asarray(angle)
        if raw.ndim != 0:
            raise ValueError("angle must be a scalar.")
        value = float(_real_array(angle, "angle"))
        return (value + np.pi) % (2.0 * np.pi) - np.pi

    def validate_state(self, x: np.ndarray) -> np.ndarray:
        """Return a detached finite state of shape (state_dim,).

        Enforce a vector rather than flattening arbitrary matrices. Lists
        and one-dimensional arrays are accepted; callers using column/row
        vectors must explicitly choose their conversion at the boundary.
        Physical bounds, angle conventions and observability remain separate
        model responsibilities; numerical validity does not establish them.
        """
        dimension = self._state_dimension()
        state = _real_array(x, f"{self.name} state")
        if state.shape != (dimension,):
            raise ValueError(f"{self.name} state must have shape ({dimension},), "
                             f"but received {state.shape}.")
        return state

    @staticmethod
    def validate_dt(dt: float) -> float:
        """Return a finite nonnegative scalar interval in seconds.

        Accept numerical Python/NumPy scalars and zero-dimensional arrays.
        Reject booleans, strings, complex values and vectors. Zero intervals
        are allowed; concrete elapsed-time models should satisfy f(x,0)=x
        modulo angular wrapping and Q(x,0)=0.
        """
        raw = np.asarray(dt)
        if raw.ndim != 0:
            raise ValueError("dt must be a real numerical scalar.")
        interval = float(_real_array(dt, "dt"))
        if interval < 0:
            raise ValueError("dt must be non-negative.")
        return interval

    def predict_state(self, x: np.ndarray, dt: float) -> np.ndarray:
        """Evaluate nominal f(x,dt) with validated, detached inputs/output.

        No process noise sample is added. For a fixed x and zero conditional
        mean disturbance this is its conditional next-state mean. It is not
        generally the mean propagated from an uncertain previous state.
        """
        state = self.validate_state(x)
        interval = self.validate_dt(dt)
        return self.validate_state(self._predict_state(state, interval))

    @abstractmethod
    def _predict_state(self, x: np.ndarray, dt: float) -> np.ndarray:
        """Implement nominal dynamics using a validated vector and interval.

        Return shape (state_dim,). Concrete angular models define the wrapping
        policy for their angular output coordinates.
        """
        raise NotImplementedError

    def process_noise(self, x: np.ndarray, dt: float) -> np.ndarray:
        """Return validated conditional state-disturbance covariance Q(x,dt).

        Entries have the product units of their two state coordinates. Q need
        only be positive semidefinite: low-dimensional driving noise often
        produces a singular covariance in the full state space.

        Tiny asymmetry is symmetrized; tiny negative eigenvalues consistent
        with the numerical tolerance are clipped. Materially invalid matrices
        are rejected rather than repaired. Tolerances depend on coordinate
        scaling and do not represent physical noise or a calibration claim.
        """
        state = self.validate_state(x)
        interval = self.validate_dt(dt)
        dimension = self._state_dimension()
        Q = _real_array(self._process_noise(state, interval),
                        f"{self.name} process-noise covariance")
        expected = (dimension, dimension)
        if Q.shape != expected:
            raise ValueError(f"{self.name} process-noise covariance must have "
                             f"shape {expected}, but received {Q.shape}.")
        if not np.allclose(Q, Q.T, rtol=0, atol=1e-10):
            raise ValueError(f"{self.name} process-noise covariance must be symmetric.")
        Q = .5 * (Q + Q.T)
        values, vectors = np.linalg.eigh(Q)
        tolerance = 1e-10 * max(1., float(np.max(np.abs(values))))
        if values.min() < -tolerance:
            raise ValueError(f"{self.name} process-noise covariance must be positive semidefinite.")
        if values.min() < 0:
            Q = (vectors * np.maximum(values, 0)) @ vectors.T
            Q = .5 * (Q + Q.T)
        return Q.copy()

    @abstractmethod
    def _process_noise(self, x: np.ndarray, dt: float) -> np.ndarray:
        """Construct conditional state-noise covariance from validated inputs.

        State-dependent noise requires a stated approximation in Gaussian
        filters, e.g. evaluation at the mean. This method does not integrate
        Q(X,dt) over an uncertain X or prescribe the filter's approximation.
        """
        raise NotImplementedError

    def jacobian(self, x: np.ndarray, dt: float) -> np.ndarray:
        """Optional transition derivative A = partial f(x,dt) / partial x.

        This is a local derivative, not a global linear transition matrix.
        Angular derivatives use a consistent local chart away from branch cuts.
        No silent finite-difference fallback is provided. An EKF-compatible
        concrete model must override this method; a UKF need not use it.
        """
        self.validate_state(x)
        self.validate_dt(dt)
        raise NotImplementedError(f"{self.name} does not provide a transition Jacobian.")

    def validate_jacobian(self, jacobian: np.ndarray) -> np.ndarray:
        """Return a detached finite real square Jacobian of the required size.

        A Jacobian need not be symmetric or positive semidefinite. Shape and
        finiteness checks do not establish derivative accuracy; compare it
        with independent numerical derivatives in concrete model tests.
        """
        dimension = self._state_dimension()
        F = _real_array(jacobian, f"{self.name} transition Jacobian")
        expected = (dimension, dimension)
        if F.shape != expected:
            raise ValueError(f"{self.name} transition Jacobian must have shape "
                             f"{expected}, but received {F.shape}.")
        return F


# FUTURE OPERATOR-BASED EXTENSION
# -------------------------------
# These nominal dynamics/covariances remain analytical benchmark components.
# Their nonlinear behavior does not by itself require or implement an operator
# representation of the belief. A learned conditional transition may describe
# richer distributions, but needs its own interface and recursive numerical
# construction. Neither f nor its Jacobian is the learned matrix acting on
# support weights or feature coefficients. State/context sufficiency, interval
# dependence, probability validity and integration accuracy remain explicit
# questions in the learned construction.
