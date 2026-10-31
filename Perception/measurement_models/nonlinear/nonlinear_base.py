"""Validated nominal geometry for conventional nonlinear observation models.

For Z=h(X)+V, h maps one specified state into observation coordinates. Under
zero conditional-mean additive noise it supplies the conditional observation
mean at that state in Euclidean coordinates. Angular coordinates require a
suitable local representation and periodic residual/likelihood conventions.

For uncertain X, E[h(X)] generally differs from h(E[X]). A filter must propagate
state uncertainty to predict observation moments: this base evaluates h only.
It does not calculate the innovation covariance, likelihood, posterior or gain.

A configured covariance R does not imply Gaussian noise. If Gaussian noise is
assumed, Z conditional on fixed X=x is Gaussian (in the stated coordinate
model), but the joint state-observation law and posterior need not be Gaussian.
"""
from abc import abstractmethod

import numpy as np

from ..measurement_base import MeasurementModel, _dimension, _real_array


class NonlinearMeasurementModel(MeasurementModel):
    """Validated observation evaluation with an optional local Jacobian.

    Concrete subclasses implement _predict_measurement and measurement_noise.
    The parent supplies state/observation validation, semantic identity checks,
    covariance validation and default residuals with declared angular wrapping.

    The public predict_measurement wrapper validates inputs and outputs.
    Subclasses use the underscore method for the physical geometry; callers
    should use the public wrapper rather than bypass its checks.

    Calibration, sensor pose, ego motion and other required context must be
    explicit in a concrete model's configuration or a deliberate interface
    extension. This no-context signature does not establish those quantities.
    Numerical admissibility does not establish observability or model accuracy.
    """

    def predict_measurement(self, x: np.ndarray) -> np.ndarray:
        """Return detached h(x) with exact shape (measurement_dim,).

        No measurement-noise sample is added. No silent matrix flattening or
        complex-to-real conversion is allowed. Concrete geometry defines where
        h is meaningful, for example visibility or zero-range singularities;
        it must handle or reject those cases explicitly.

        Angular outputs are not automatically wrapped by this wrapper. Concrete
        models choose their output convention; residual handles differences.
        """
        state = self.validate_state(x)
        prediction = self._predict_measurement(state)
        return self.validate_measurement(prediction)

    @abstractmethod
    def _predict_measurement(self, x: np.ndarray) -> np.ndarray:
        """Implement h(x) using a validated, detached native-state vector.

        Return a finite real vector in the declared observation order/units.
        The supplied state is a copy; modifying it must not change filter state.
        """
        raise NotImplementedError

    @abstractmethod
    def measurement_noise(self) -> np.ndarray:
        """Return configured observation-noise covariance R, not innovation S.

        Shape is (measurement_dim, measurement_dim). R must be finite, real,
        symmetric and PSD; call inherited validate_covariance before returning
        it. Correlated components and singular covariances are allowed.

        This abstract method is not automatically wrapped. State-dependent
        noise, bias or nonadditive noise require an explicit model/filter
        extension; simply providing a nonlinear h does not cover them.
        """
        raise NotImplementedError

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """Optional derivative J_h(x)=partial h(x)/partial x.

        Shape is (measurement_dim, state_dim). An EKF-compatible model must
        override this method, validate its state and call validate_jacobian on
        the result. A UKF can propagate observations without this derivative.
        There is no silent numerical differentiation fallback.

        The derivative is local; angular components use a consistent local
        chart away from branch cuts. Domain singularities need model-specific
        treatment. Supplying a Jacobian does not establish observability.
        """
        self.validate_state(x)
        raise NotImplementedError(f'{self.name} does not provide a measurement Jacobian.')

    def validate_jacobian(self, jacobian: np.ndarray) -> np.ndarray:
        """Validate and detach a finite real rectangular observation Jacobian.

        J need not be square, symmetric, PSD or full rank. These checks establish
        dimensions and numerical validity, not whether J differentiates h
        correctly. Concrete-model tests should compare against independent
        numerical derivatives, using wrapped angular differences where needed.
        """
        nx = _dimension(self.state_dim, 'state_dim')
        nz = _dimension(self.measurement_dim, 'measurement_dim')
        J = _real_array(jacobian, f'{self.name} observation Jacobian')
        expected = (nz, nx)
        if J.shape != expected:
            raise ValueError(f'{self.name} Jacobian must have shape {expected}, '
                             f'but received {J.shape}.')
        return J


# FUTURE OPERATOR-BASED EXTENSION
# -------------------------------
# A learned observation law can describe more than nominal h and covariance R.
# Its interface must support the likelihood/conditional probabilities required
# by the chosen recursive belief construction. Observation learning is distinct
# from transition learning and from directly learning a posterior. Preserve
# state/observation coordinate definitions, context, association and detection
# conditioning when constructing paired training data. Bayesian conditioning
# remains necessary after the learned observation model is evaluated.
