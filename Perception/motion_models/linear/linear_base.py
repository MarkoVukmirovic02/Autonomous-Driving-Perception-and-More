"""Common interface for conventional linear motion models.

PURPOSE
-------
A concrete motion model supplies the nominal transition matrix F(dt) and
process-disturbance covariance Q(dt). A filter uses them to propagate a
belief; the motion model itself does not store or update that belief.

For the zero-mean additive-disturbance model:

    x_k = F(dt) x_{k-1} + w_k,
    E[w_k] = 0,  Cov(w_k) = Q(dt),

independence (or the corresponding zero cross-covariance) between the
previous estimation error and w_k gives the moment prediction:

    m_minus = F @ m_plus,
    P_minus = F @ P_plus @ F.T + Q.

A Gaussian initial belief and independent Gaussian disturbances make this
an exact Gaussian prediction. F and Q alone do not establish Gaussianity
or temporal independence. They also do not specify a general non-Gaussian
transition distribution completely.

This interface represents homogeneous linear state transitions: there is
no explicit affine offset or known-control term B @ u. A future extension
must specify such inputs explicitly rather than hiding them in Q.
"""
from abc import ABC, abstractmethod

import numpy as np


class LinearMotionModel(ABC):
    """Required operations for a concrete linear motion model.

    ABC and abstractmethod prevent instantiation until a subclass implements
    both matrix methods. This is explicit inheritance, unlike a Protocol's
    structural compatibility.

    Subclasses supply:
        name: descriptive model name for configuration and diagnostics.
        state_dim: positive integer giving the native state-vector dimension.

    The annotations below do not create or validate those attributes.
    State ordering, coordinate conventions and noise-parameter units must
    be documented by each concrete model.

    Both methods must return finite real arrays of shape
    (state_dim, state_dim). Q must be symmetric positive semidefinite;
    it need not be positive definite. Rank-deficient process noise is valid.

    Responsibility for checks:
        * Concrete matrix methods call validate_dt before using the interval.
        * Concrete constructors validate their dimensions/noise parameters.
        * Consuming filters validate returned matrix shapes and admissibility
          at their numerical boundary, including for third-party models.

    This base class does not intercept subclass methods or automatically
    validate their results. It also imposes no Gaussian-noise requirement.
    """

    name: str
    state_dim: int

    @staticmethod
    def validate_dt(dt: float) -> float:
        """Return a finite, nonnegative numerical scalar interval in seconds.

        Accept Python/NumPy integer and floating-point scalars, including
        zero-dimensional numerical arrays. Reject booleans, complex values,
        strings and vectors rather than relying on float's coercion rules.

        Zero is allowed for explicit same-time operations. For an ordinary
        elapsed-time model, F(0)=I and Q(0)=0 are expected and should be tested
        in the concrete model. This helper cannot enforce those identities.
        """
        raw = np.asarray(dt)
        if raw.ndim != 0 or raw.dtype.kind not in 'iuf':
            raise ValueError("dt must be a real numerical scalar.")
        try:
            interval = float(raw)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("dt must be a finite real numerical scalar.") from error
        if not np.isfinite(interval):
            raise ValueError("dt must be finite.")
        if interval < 0.0:
            raise ValueError("dt must be non-negative.")
        return interval

    @abstractmethod
    def transition_matrix(self, dt: float) -> np.ndarray:
        """Return F(dt), shape (state_dim, state_dim).

        The matrix maps the previous native state to its nominal next state.
        It does not propagate a covariance or generate random disturbances.
        The implementation must validate dt explicitly with validate_dt.
        """
        raise NotImplementedError

    @abstractmethod
    def process_noise(self, dt: float) -> np.ndarray:
        """Return Q(dt), shape (state_dim, state_dim).

        Q is the covariance of the additive STATE disturbance accumulated
        over this interval, not necessarily the covariance of an underlying
        acceleration/jerk input. The concrete model defines the disturbance
        construction and its sampling-interval dependence.

        Entry (i,j) has units equal to state-coordinate i units multiplied
        by state-coordinate j units. A single matrix may therefore contain
        entries with different physical units.

        Returning Q does not sample noise or enforce independence assumptions.
        The implementation must validate dt explicitly with validate_dt.
        """
        raise NotImplementedError


# FUTURE OPERATOR-BASED EXTENSION
# -------------------------------
# Keep this interface as the conventional benchmark's linear motion contract.
# A learned transition law q(x_next | x_previous, context) generally requires
# a different interface, chosen alongside the numerical belief representation.
#
# A matrix acting on support probability weights or basis coefficients is not
# the same object as F, which acts on physical-state coordinates. Such a learned
# operator may propagate a richer belief without Gaussian moment closure.
# Its dimensions, probability validity, reference measure and time-step/context
# dependence must be specified in that separate construction.
#
# Learning a transition model does not remove Bayesian prediction: it supplies
# a learned representation used to carry out or approximate that operation.
