"""Validated linear observation geometry for conventional filters.

For Z=H X+V, H maps native state coordinates to observation coordinates.
It may select some state components, combine them or change their units.
It need not be square, invertible or full rank.

This module supplies h(x)=H@x and its exact Jacobian H. A concrete model
supplies H and observation-noise covariance R. Neither matrix alone makes
observations Gaussian: that requires additional probabilistic assumptions.

This interface has no explicit affine offset b in h(x)=H@x+b. If needed,
introduce that term explicitly rather than treating a systematic offset as
zero-mean noise. Geometry/calibration must be specified by concrete models.
"""
from abc import abstractmethod

import numpy as np

from ..measurement_base import MeasurementModel, _dimension, _real_array


class LinearMeasurementModel(MeasurementModel):
    """Base for models whose observation is linear in the native state.

    A subclass supplies measurement_matrix() and measurement_noise(). The
    inherited MeasurementModel contract defines dimensions, observation
    semantics and residual handling. It also supplies validate_covariance,
    which concrete measurement_noise methods should use when returning R.

    H is independent of x for a fixed model configuration. This no-argument
    interface can be reconfigured externally, but does not itself manage
    changing sensor context or synchronize it with capture timestamps.

    Linear image-box selection is an example. A camera projection of a
    physical world state is generally nonlinear and belongs in a different
    concrete observation model.
    """

    @abstractmethod
    def measurement_matrix(self) -> np.ndarray:
        """Return real finite H, shape (measurement_dim, state_dim).

        Each row defines one observation coordinate; columns follow native
        state ordering. The entry H[i,j] converts state-coordinate j into
        observation-coordinate i and therefore carries the corresponding
        units ratio where applicable.

        Public prediction/Jacobian methods validate and detach the matrix.
        Calling this subclass method directly does not invoke those wrappers.
        """
        raise NotImplementedError

    def _validated_measurement_matrix(self) -> np.ndarray:
        """Shared H validation; reject complex values before float conversion."""
        nx = _dimension(self.state_dim, 'state_dim')
        nz = _dimension(self.measurement_dim, 'measurement_dim')
        H = _real_array(self.measurement_matrix(), f'{self.name} measurement matrix')
        expected = (nz, nx)
        if H.shape != expected:
            raise ValueError(f'{self.name} measurement matrix must have shape '
                             f'{expected}, but received {H.shape}.')
        # No symmetry/PSD requirement: H is a map, not a covariance.
        return H

    def predict_measurement(self, x: np.ndarray) -> np.ndarray:
        """Return detached nominal observation h(x)=H@x without adding noise.

        For uncertain X and a fixed H, E[H@X]=H@E[X] exactly when the mean
        exists. This differs from general nonlinear observation propagation.
        The filter separately calculates S=H@P@H.T+R under the relevant
        zero conditional-mean/cross-covariance assumptions.

        Angular output values are not automatically wrapped here. The
        inherited residual handles declared angular differences locally;
        periodic likelihoods and angular charts still require care.
        """
        state = self.validate_state(x)
        H = self._validated_measurement_matrix()
        try:
            with np.errstate(over='raise', invalid='raise'):
                prediction = H @ state
        except FloatingPointError as error:
            raise ValueError('Linear observation prediction exceeds numerical range.') from error
        return self.validate_measurement(prediction)

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """Return the exact derivative partial h/partial x=H.

        x is validated for interface consistency even though this derivative
        does not depend on its value. Returning a copy isolates caller edits
        from a concrete model's stored matrix. This is a rectangular derivative,
        not the state-transition matrix or the Kalman gain.
        """
        self.validate_state(x)
        return self._validated_measurement_matrix()


# FUTURE OPERATOR-BASED EXTENSION
# -------------------------------
# An analytical linear likelihood can remain a controlled observation model
# while the transition law or belief representation is learned. Learning an
# observation distribution is a separate experiment: H and R would not in
# general capture the resulting conditional law. A learned likelihood operator
# must not be confused with H or with the gain computed from a predicted belief.
