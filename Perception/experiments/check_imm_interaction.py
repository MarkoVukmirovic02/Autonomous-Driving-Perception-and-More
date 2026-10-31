import numpy as np

from Perception.tracking.types import (
    GaussianState,
    MeasurementSpaceIdentity,
)

from Perception.tracking.imm import IMM
from Perception.tracking.imm_adapter import IMMFilterAdapter


class TestEstimator:
    """Minimal native-state container used only to test interaction."""

    measurement_angle_indices = ()

    measurement_space_identity = MeasurementSpaceIdentity(
        name="interaction_test_observation",
        coordinates=("test_value",),
        dimension=1,
    )

    def __init__(self, x, P, angle_indices=()):
        self.x = np.array(x, dtype=float)
        self.P = np.array(P, dtype=float)
        self.state_angle_indices = tuple(angle_indices)

    def state_snapshot(self):
        return GaussianState(self.x, self.P)



def make_test_adapter(x, P, *, angle_indices=()):
    estimator = TestEstimator(x, P, angle_indices)

    def identity_conversion(x, P):
        return GaussianState(x, P)

    return IMMFilterAdapter(
        estimator,
        to_common=identity_conversion,
        to_native=identity_conversion,
        state_space=(
            "interaction_test",
            len(x),
            tuple(angle_indices),
        ),
    )


def assert_valid_covariance(P):
    assert np.all(np.isfinite(P))

    np.testing.assert_allclose(
        P, P.T, rtol=0.0, atol=1e-10
    )

    eigenvalues = np.linalg.eigvalsh(P)
    tolerance = 1e-10 * max(1.0, np.max(np.abs(eigenvalues)))
    assert eigenvalues.min() >= -tolerance


def check_probabilities():
    imm = IMM(
        filters=[
            make_test_adapter([0.], [[1.]]),
            make_test_adapter([2.], [[1.]]),
        ],
        mode_probabilities=[0.5, 0.5],
        transition_matrix=[
            [0.95, 0.05],
            [0.10, 0.90],
        ],
    )

    predicted = imm.predict_mode_probabilities()
    weights = imm.mixing_probabilities(predicted)

    np.testing.assert_allclose(predicted, [0.525, 0.475])

    expected = np.array([
        [0.475 / 0.525, 0.025 / 0.475],
        [0.050 / 0.525, 0.450 / 0.475],
    ])

    np.testing.assert_allclose(weights, expected)
    np.testing.assert_allclose(weights.sum(axis=0), [1., 1.])

    print("PASS: predicted probabilities and mixing weights")


def check_mixed_moments():
    imm = IMM(
        filters=[
            make_test_adapter([0.], [[1.]]),
            make_test_adapter([2.], [[1.]]),
        ],
        mode_probabilities=[0.5, 0.5],
        transition_matrix=np.full((2, 2), 0.5),
    )

    before = [a.state_snapshot() for a in imm.filters]

    predicted = imm.predict_mode_probabilities()
    weights = imm.mixing_probabilities(predicted)
    mixed = imm.mix_states(weights)

    for state in mixed:
        np.testing.assert_allclose(state.x, [1.])
        np.testing.assert_allclose(state.P, [[2.]])
        assert_valid_covariance(state.P)

    # Calculating interaction must not mutate the estimators.
    for adapter, original in zip(imm.filters, before):
        current = adapter.state_snapshot()
        np.testing.assert_array_equal(current.x, original.x)
        np.testing.assert_array_equal(current.P, original.P)

    print("PASS: mixed moments and no mutation")

    # Identical means: only within-component uncertainty remains.
    imm = IMM(
        filters=[
            make_test_adapter([3.], [[1.]]),
            make_test_adapter([3.], [[9.]]),
        ],
        mode_probabilities=[0.25, 0.75],
        transition_matrix=np.full((2, 2), 0.5),
    )

    predicted = imm.predict_mode_probabilities()
    mixed = imm.mix_states(
        imm.mixing_probabilities(predicted)
    )

    for state in mixed:
        np.testing.assert_allclose(state.x, [3.])
        np.testing.assert_allclose(state.P, [[7.]])

    # Zero component covariance: disagreement still creates uncertainty.
    imm = IMM(
        filters=[
            make_test_adapter([0.], [[0.]]),
            make_test_adapter([2.], [[0.]]),
        ],
        mode_probabilities=[0.5, 0.5],
        transition_matrix=np.full((2, 2), 0.5),
    )

    predicted = imm.predict_mode_probabilities()
    mixed = imm.mix_states(
        imm.mixing_probabilities(predicted)
    )

    for state in mixed:
        np.testing.assert_allclose(state.x, [1.])
        np.testing.assert_allclose(state.P, [[1.]])

    print("PASS: identical means and disagreement-only uncertainty")


def check_angles():
    component_variance = 0.01

    imm = IMM(
        filters=[
            make_test_adapter(
                [np.deg2rad(179.)],
                [[component_variance]],
                angle_indices=(0,),
            ),
            make_test_adapter(
                [np.deg2rad(-179.)],
                [[component_variance]],
                angle_indices=(0,),
            ),
        ],
        mode_probabilities=[0.5, 0.5],
        transition_matrix=np.full((2, 2), 0.5),
    )

    predicted = imm.predict_mode_probabilities()
    mixed = imm.mix_states(
        imm.mixing_probabilities(predicted)
    )

    expected_variance = component_variance + np.deg2rad(1.)**2

    for state in mixed:
        # The mean must be at the +/-pi boundary, not near zero.
        np.testing.assert_allclose(
            abs(state.x[0]), np.pi, atol=1e-12
        )
        np.testing.assert_allclose(
            state.P, [[expected_variance]], atol=1e-12
        )
        assert_valid_covariance(state.P)

    print("PASS: circular mean and wrapped covariance differences")

    # Opposite, equally weighted headings have no unique circular mean.
    ambiguous = IMM(
        filters=[
            make_test_adapter([0.], [[0.01]], angle_indices=(0,)),
            make_test_adapter([np.pi], [[0.01]], angle_indices=(0,)),
        ],
        mode_probabilities=[0.5, 0.5],
        transition_matrix=np.full((2, 2), 0.5),
    )

    predicted = ambiguous.predict_mode_probabilities()
    weights = ambiguous.mixing_probabilities(predicted)

    try:
        ambiguous.mix_states(weights)
    except ValueError as error:
        assert "ambiguous" in str(error).lower()
    else:
        raise AssertionError("Ambiguous headings should be rejected.")

    print("PASS: ambiguous circular mean rejection")


def check_unreachable_mode():
    imm = IMM(
        filters=[
            make_test_adapter([0.], [[1.]]),
            make_test_adapter([9.], [[4.]]),
        ],
        mode_probabilities=[1., 0.],
        transition_matrix=np.eye(2),
    )

    predicted = imm.predict_mode_probabilities()
    weights = imm.mixing_probabilities(predicted)
    mixed = imm.mix_states(weights)

    np.testing.assert_array_equal(predicted, [1., 0.])
    np.testing.assert_array_equal(weights, np.eye(2))

    # Unreachable destination retains its own starting distribution.
    np.testing.assert_array_equal(mixed[1].x, [9.])
    np.testing.assert_array_equal(mixed[1].P, [[4.]])

    # Even a larger finite likelihood cannot revive a zero prior.
    posterior = imm.posterior_mode_probabilities(
        np.array([-10., 0.])
    )
    np.testing.assert_array_equal(posterior, [1., 0.])

    print("PASS: unreachable-mode fallback and zero posterior support")


def main():
    check_probabilities()
    check_mixed_moments()
    check_angles()
    check_unreachable_mode()

    print("All interaction checks passed.")


if __name__ == "__main__":
    main()