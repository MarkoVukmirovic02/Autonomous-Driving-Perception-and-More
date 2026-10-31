"""Heterogeneous IMM using existing filters and native-coordinate interaction.

Measurement diagnostics use moment-matched gating distance and a component-
mixture log density. Prediction/update failures after mutation poison the bank;
restore a native checkpoint or rebuild before reuse. No hidden rollback claim.
"""
from dataclasses import dataclass

from .types import (
    GaussianState,
    MeasurementSpaceIdentity,
    validate_gaussian_state,
    MeasurementMoments,
    InnovationDiagnostics,
    Measurement,
)
import numpy as np


@dataclass(frozen=True)
class IMMCheckpoint:
    """Detached native beliefs and mode probabilities, not sensor/time context.

    Restore requires the same bank/configuration. External provider state and
    clocks are not included. Restore clears diagnostic caches through each
    filter's public replace_state contract.
    """
    states: tuple[GaussianState, ...]
    probabilities: np.ndarray
    signature: tuple

    def __post_init__(self):
        states = tuple(GaussianState(s.x, s.P) for s in self.states)
        for state in states:
            state.x.setflags(write=False)
            state.P.setflags(write=False)
        probabilities = np.array(self.probabilities, dtype=float, copy=True)
        probabilities.setflags(write=False)
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "probabilities", probabilities)


class IMM:
    def __init__(self,filters,mode_probabilities,transition_matrix):
        self.filters = list(filters)
        if np.iscomplexobj(mode_probabilities) or np.iscomplexobj(transition_matrix):
            raise ValueError("probabilities and transitions must be real")
        self.mode_probabilities = np.array(
            mode_probabilities, dtype=float, copy=True
        )
        self.transition_matrix = np.array(
            transition_matrix, dtype=float, copy=True
        )

        self._failed = False
        self._validate_configuration()
        self._native_dimensions = tuple(a.state_snapshot().x.size for a in self.filters)

    @property
    def failed(self):
        return self._failed

    def _ensure_healthy(self):
        if self._failed:
            raise RuntimeError("IMM bank is invalid after a partial failure; "
                               "restore a checkpoint or rebuild the bank")

    @property
    def coordinate_frame(self):
        return self.filters[0].coordinate_frame

    @property
    def measurement_frame(self):
        return self.filters[0].measurement_frame

    @property
    def common_space(self):
        return self.filters[0].common_space

    def _signature(self):
        modes = tuple((a.state_space, a.common_space, a.coordinate_frame,
                       a.measurement_frame, a.state_angle_indices,
                       a.measurement_space_identity, a.measurement_context,
                       id(a.filter), id(a.to_common), id(a.to_native),
                       a.minimum_speed) for a in self.filters)
        return modes, tuple(map(tuple, self.transition_matrix))

    def checkpoint(self):
        self._ensure_healthy()
        return IMMCheckpoint(tuple(a.state_snapshot() for a in self.filters),
                             self.mode_probabilities, self._signature())

    def restore_checkpoint(self, checkpoint):
        if not isinstance(checkpoint, IMMCheckpoint):
            raise TypeError("expected IMMCheckpoint")
        if checkpoint.signature != self._signature():
            raise ValueError("checkpoint belongs to a different bank/configuration")
        if len(checkpoint.states) != len(self.filters):
            raise ValueError("checkpoint mode count mismatch")
        probabilities = self._validated_probabilities(checkpoint.probabilities)
        states = [validate_gaussian_state(s.x, s.P, dimension, a.state_angle_indices)
                  for s, a, dimension in zip(checkpoint.states, self.filters,
                                              self._native_dimensions)]
        try:
            for adapter, state in zip(self.filters, states):
                adapter.replace_state(state.x, state.P)
            self.mode_probabilities = probabilities
        except Exception:
            self._failed = True
            raise
        self._failed = False
        return self.state_snapshot()

    def _validated_probabilities(self, values):
        values = np.asarray(values)
        if np.iscomplexobj(values):
            raise ValueError("mode probabilities must be real")
        values = np.asarray(values, dtype=float)
        if (values.shape != (len(self.filters),) or not np.all(np.isfinite(values))
                or np.any(values < 0)
                or not np.isclose(values.sum(), 1., rtol=0, atol=1e-10)):
            raise ValueError("mode probabilities must be nonnegative and sum to one")
        return values.copy()

    def set_transition_matrix(self, values):
        """Install a known per-step matrix; no transition learning is implied.

        For variable sampling intervals, the caller may supply a matrix derived
        from a continuous-time mode generator. A fixed matrix is per call with
        positive dt, not per second.
        """
        self._ensure_healthy()
        raw = np.asarray(values)
        if np.iscomplexobj(raw):
            raise ValueError("transition probabilities must be real")
        values = np.asarray(raw, dtype=float)
        count = len(self.filters)
        if (values.shape != (count, count) or not np.all(np.isfinite(values))
                or np.any(values < 0)
                or not np.allclose(values.sum(axis=1), 1., rtol=0, atol=1e-10)):
            raise ValueError("transition matrix must be row stochastic")
        self.transition_matrix = values.copy()

    @staticmethod
    def _validated_dt(dt):
        raw = np.asarray(dt)
        if raw.ndim != 0 or raw.dtype.kind not in 'iuf':
            raise ValueError("dt must be a real nonnegative scalar")
        dt = float(raw)
        if not np.isfinite(dt) or dt < 0:
            raise ValueError("dt must be finite and nonnegative")
        return dt

    def update_measurement(self, measurement):
        """Numerical managed update; wrapper owns timestamp/sensor checks."""
        if not isinstance(measurement, Measurement):
            raise TypeError("expected Measurement")
        if measurement.measurement_space_identity != self.measurement_space_identity:
            raise ValueError("measurement must declare the matching observation space")
        if (measurement.coordinate_frame is not None and self.measurement_frame is not None
                and measurement.coordinate_frame != self.measurement_frame):
            raise ValueError("measurement frame mismatch")
        return self.update(measurement.value, R=measurement.covariance)

    @property
    def state_angle_indices(self):
        # The common Cartesian output contains no angular coordinates.
        return ()

    def state_snapshot(self):
        mean, covariance = self.combine_estimates()

        return validate_gaussian_state(
            mean,
            covariance,
            mean.size,
            self.state_angle_indices,
        )

    @property
    def x(self):
        return self.state_snapshot().x

    @property
    def P(self):
        return self.state_snapshot().P

    @property
    def measurement_space_identity(self):
        return self.filters[0].measurement_space_identity

    @property
    def measurement_angle_indices(self):
        return self.filters[0].measurement_angle_indices


    def _validate_measurement_spaces(self):
        expected = self.filters[0].measurement_space_identity

        if not isinstance(expected, MeasurementSpaceIdentity):
            raise ValueError(
                "Every IMM mode must declare a measurement-space identity."
            )

        for index, adapter in enumerate(self.filters):
            identity = adapter.measurement_space_identity

            if identity != expected:
                raise ValueError(
                    f"Mode {index} uses a different measurement space."
                )

            if adapter.measurement_context != self.filters[0].measurement_context:
                raise ValueError(f"Mode {index} uses different sensor geometry/context")

            if adapter.measurement_angle_indices != expected.angle_indices:
                raise ValueError(
                    f"Mode {index} has inconsistent measurement angles."
                )



    def _validate_configuration(self):
        number_of_models = len(self.filters)

        if number_of_models == 0:
            raise ValueError("IMM requires at least one filter.")

        if self.mode_probabilities.shape != (number_of_models,):
            raise ValueError(
                    "Provide one mode probability per filter."
                )

        if self.transition_matrix.shape != (
                number_of_models, number_of_models
            ):
            raise ValueError(
                    "Transition matrix must have shape (M, M)."
                )

        if (
                not np.all(np.isfinite(self.mode_probabilities))
                or np.any(self.mode_probabilities < 0)
            ):
            raise ValueError(
                    "Mode probabilities must be finite and nonnegative."
                )

        if not np.isclose(
                self.mode_probabilities.sum(), 1.0,
                rtol=0.0, atol=1e-10
            ):
            raise ValueError("Mode probabilities must sum to one.")

        if (
                not np.all(np.isfinite(self.transition_matrix))
                or np.any(self.transition_matrix < 0)
            ):
            raise ValueError(
                    "Transition probabilities must be finite "
                    "and nonnegative."
                )

        if not np.allclose(
                self.transition_matrix.sum(axis=1), 1.0,
                rtol=0.0, atol=1e-10
            ):
            raise ValueError(
                    "Each transition-matrix row must sum to one."
                )

        first = self.filters[0]
        for adapter in self.filters:
            if adapter.common_space != first.common_space:
                raise ValueError("all adapters must declare the same common state space")
            if adapter.coordinate_frame != first.coordinate_frame:
                raise ValueError("all modes must use the same state frame")
            if adapter.measurement_frame != first.measurement_frame:
                raise ValueError("all modes must use the same observation frame")
        common_states = [adapter.common_state() for adapter in self.filters]
        dimension = common_states[0].x.size
        if any(state.x.shape != (dimension,) for state in common_states):
            raise ValueError("common state dimensions do not match")
        if first.common_space is not None and len(first.common_space) != dimension:
            raise ValueError("common state identity does not match dimension")

        underlying_filters = [adapter.filter for adapter in self.filters]

        if len({id(f) for f in underlying_filters}) != number_of_models:
            raise ValueError(
                "Each mode must use a distinct underlying filter."
            )

        for adapter in self.filters:
            state = adapter.state_snapshot()
            dimension = state.x.size
            validate_gaussian_state(state.x, state.P, dimension, adapter.state_angle_indices)
            for other in self.filters:
                if adapter.state_space == other.state_space:
                    if (other.state_snapshot().x.size != dimension
                            or other.state_angle_indices != adapter.state_angle_indices):
                        raise ValueError("matching native identities have inconsistent layouts")

            if state.x.shape != (dimension,):
                raise ValueError("Native means must be 1D vectors.")

            if state.P.shape != (dimension, dimension):
                raise ValueError(
                    "Each covariance must match its native state."
                )
        self._validate_measurement_spaces()



    def predict_mode_probabilities(self):
        self._ensure_healthy()
        self._validated_probabilities(self.mode_probabilities)
        mode_pred_prob = self.transition_matrix.T @ self.mode_probabilities
        return mode_pred_prob

    def mixing_probabilities(self, predicted_probabilities):
        joint_probabilities = (
            self.mode_probabilities[:, None] * self.transition_matrix
        )

        weights = np.zeros_like(joint_probabilities)

        np.divide(
            joint_probabilities,
            predicted_probabilities[None, :],
            out=weights,
            where=predicted_probabilities[None, :] > 0
        )

        unreachable = np.flatnonzero(
            predicted_probabilities == 0
        )
        weights[unreachable, unreachable] = 1.0

        return weights


    def mix_states(self, weights):
        self._ensure_healthy()
        snapshots = [
            adapter.state_snapshot()
            for adapter in self.filters
        ]

        mixed_states = []

        for j, destination in enumerate(self.filters):
            sources = np.flatnonzero(weights[:, j] > 0)

            # Preserve self-only interaction without any conversion.
            if sources.size == 1 and sources[0] == j:
                state = snapshots[j]
                mixed_states.append(GaussianState(state.x, state.P))
                continue

            components = [
                destination.convert_from(
                    self.filters[i],
                    snapshots[i],
                    destination_state=snapshots[j],
                )
                for i in sources
            ]

            destination_weights = weights[sources, j]
            points = np.stack([state.x for state in components])
            mean = destination_weights @ points

            angles = destination.state_angle_indices

            for index in angles:
                sine = destination_weights @ np.sin(points[:, index])
                cosine = destination_weights @ np.cos(points[:, index])

                if np.hypot(sine, cosine) <= 1e-12:
                    raise ValueError(
                        "Mixed heading is ambiguous: "
                        "a single Gaussian is unsuitable."
                    )

                mean[index] = np.arctan2(sine, cosine)

            differences = points - mean

            for index in angles:
                differences[:, index] = (
                    differences[:, index] + np.pi
                ) % (2 * np.pi) - np.pi

            covariance = np.zeros_like(components[0].P)

            for weight, state, difference in zip(
                destination_weights, components, differences
            ):
                covariance += weight * (
                    state.P + np.outer(difference, difference)
                )

            mixed_states.append(
                validate_gaussian_state(
                    mean,
                    covariance,
                    mean.size,
                    angles,
                )
            )

        return mixed_states

    def predict(self, dt):
        self._ensure_healthy()
        dt = self._validated_dt(dt)
        self._validate_configuration()
        # A zero-duration call must not apply another discrete mode transition.
        if dt == 0:
            return
        predicted_probabilities = self.predict_mode_probabilities()
        weights = self.mixing_probabilities(predicted_probabilities)
        mixed_states = self.mix_states(weights)
        try:
            for adapter, state in zip(self.filters, mixed_states):
                adapter.replace_state(state.x, state.P)
            for adapter in self.filters:
                adapter.predict(dt)
                adapter.state_snapshot()
            self.mode_probabilities = predicted_probabilities
        except Exception:
            self._failed = True
            raise

    def measurement_log_likelihoods(self, z, R=None):
        self._ensure_healthy()
        log_likelihoods = np.empty(len(self.filters), dtype=float)

        for j, filter_ in enumerate(self.filters):
            diagnostics = filter_.innovation_diagnostics(z, R=R)
            log_likelihoods[j] = diagnostics.log_likelihood

        return log_likelihoods


    def posterior_mode_probabilities(self, log_likelihoods):
        self._ensure_healthy()
        self._validated_probabilities(self.mode_probabilities)
        log_likelihoods = np.asarray(log_likelihoods, dtype=float)

        if log_likelihoods.shape != self.mode_probabilities.shape:
            raise ValueError(
                "Provide one log-likelihood per mode."
            )

        # -inf is allowed: it represents zero likelihood.
        if (
            np.any(np.isnan(log_likelihoods))
            or np.any(np.isposinf(log_likelihoods))
        ):
            raise ValueError(
                "Log-likelihoods must be finite or negative infinity."
            )

        log_prior = np.full_like(self.mode_probabilities, -np.inf)

        np.log(
            self.mode_probabilities,
            out=log_prior,
            where=self.mode_probabilities > 0,
        )

        log_weights = log_prior + log_likelihoods
        maximum = np.max(log_weights)

        if not np.isfinite(maximum):
            raise ValueError(
                "No mode has a finite posterior weight."
            )

        weights = np.exp(log_weights - maximum)

        return weights / weights.sum()


    def update(self, z, R=None):
        self._ensure_healthy()
        self._validate_configuration()
        log_likelihoods = self.measurement_log_likelihoods(z, R=R)
        posterior_probabilities = self.posterior_mode_probabilities(log_likelihoods)
        try:
            for adapter in self.filters:
                adapter.update(z, R=R)
                adapter.state_snapshot()
            self.mode_probabilities = posterior_probabilities
        except Exception:
            self._failed = True
            raise

    def combine_estimates(self):
        self._ensure_healthy()
        self._validated_probabilities(self.mode_probabilities)
        active = np.flatnonzero(self.mode_probabilities > 0)
        probabilities = self.mode_probabilities[active]
        states = [self.filters[j].common_state() for j in active]

        means = np.stack([state.x for state in states])
        combined_mean = probabilities @ means
        combined_covariance = np.zeros_like(states[0].P)

        for probability, state in zip(
            probabilities, states
        ):
            difference = state.x - combined_mean

            combined_covariance += probability * (
                state.P + np.outer(difference, difference)
            )

        return combined_mean, combined_covariance

    def measurement_prediction(self, R=None):
        self._ensure_healthy()
        self._validated_probabilities(self.mode_probabilities)
        self._validate_measurement_spaces()
        # Ignore zero-weight modes when calculating mixture moments.
        active = np.flatnonzero(self.mode_probabilities > 0)
        weights = self.mode_probabilities[active]

        predictions = [
            self.filters[j].measurement_prediction(R=R)
            for j in active
        ]

        means = np.stack([
            prediction.mean for prediction in predictions
        ])

        mean = weights @ means
        angles = self.measurement_angle_indices

        for index in angles:
            sine = weights @ np.sin(means[:, index])
            cosine = weights @ np.cos(means[:, index])

            if np.hypot(sine, cosine) <= 1e-12:
                raise ValueError(
                    "Predicted measurement angle is ambiguous."
                )

            mean[index] = np.arctan2(sine, cosine)

        differences = means - mean

        for index in angles:
            differences[:, index] = (
                differences[:, index] + np.pi
            ) % (2 * np.pi) - np.pi

        covariance = np.zeros_like(predictions[0].covariance)

        for weight, prediction, difference in zip(
            weights, predictions, differences
        ):
            covariance += weight * (
                prediction.covariance
                + np.outer(difference, difference)
            )

        covariance = 0.5 * (covariance + covariance.T)

        return MeasurementMoments(mean, covariance)

    def innovation_diagnostics(self, z, R=None):
        self._ensure_healthy()
        prediction = self.measurement_prediction(R=R)

        raw = np.asarray(z)

        if np.iscomplexobj(raw):
            raise ValueError("Measurement must be real.")

        z = np.asarray(raw, dtype=float)

        if (
            z.shape != prediction.mean.shape
            or not np.all(np.isfinite(z))
        ):
            raise ValueError(
                "Measurement must be finite and match "
                "the predicted observation shape."
            )

        residual = z - prediction.mean

        for index in self.measurement_angle_indices:
            residual[index] = (
                residual[index] + np.pi
            ) % (2 * np.pi) - np.pi

        # S = L L.T; solve instead of explicitly inverting S.
        L = np.linalg.cholesky(prediction.covariance)
        whitened_residual = np.linalg.solve(L, residual)
        nis = float(whitened_residual @ whitened_residual)

        active = np.flatnonzero(self.mode_probabilities > 0)
        log_terms = np.empty(active.size, dtype=float)

        for position, j in enumerate(active):
            diagnostics = self.filters[j].innovation_diagnostics(
                z, R=R
            )
            log_likelihood = diagnostics.log_likelihood

            if np.isnan(log_likelihood) or np.isposinf(log_likelihood):
                raise ValueError(
                    "Component log-likelihood must be finite "
                    "or negative infinity."
                )

            log_terms[position] = (
                np.log(self.mode_probabilities[j])
                + log_likelihood
            )

        # Stable log(sum(exp(log_terms))); permits an all-zero density.
        mixture_log_likelihood = float(
            np.logaddexp.reduce(log_terms)
        )

        return InnovationDiagnostics(
            residual=residual,
            covariance=prediction.covariance,
            nis=nis,
            log_likelihood=mixture_log_likelihood,
            whitened_residual=whitened_residual,
        )