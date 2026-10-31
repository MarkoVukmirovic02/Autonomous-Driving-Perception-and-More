import numpy as np

from Perception.experiments.imm_setup import build_imm
from Perception.tracking.types import GaussianState
from Perception.tracking.state_conversion import (
    common_to_ctrv_low_speed,
    common_to_ctra_low_speed,
)


def assert_valid_covariance(P):
    assert np.all(np.isfinite(P))

    np.testing.assert_allclose(
        P, P.T, rtol=0.0, atol=1e-10
    )

    eigenvalues = np.linalg.eigvalsh(P)
    tolerance = 1e-10 * max(1.0, np.max(np.abs(eigenvalues)))
    assert eigenvalues.min() >= -tolerance


def check_fallback(conversion, include_acceleration):
    common_mean = np.array([10., 20., 0.1, 0.05, 0.2, -0.1])
    common_covariance = np.diag([1., 2., 0.04, 0.09, 0.16, 0.25])

    if include_acceleration:
        native_mean = np.array([0., 0., 8., 0., 0.8, 0.15])
        yaw_index, rate_index = 4, 5
    else:
        native_mean = np.array([0., 0., 8., 0.8, 0.15])
        yaw_index, rate_index = 3, 4

    native_covariance = np.eye(native_mean.size)
    native_covariance[yaw_index, yaw_index] = 0.04
    native_covariance[rate_index, rate_index] = 0.01
    native_covariance[yaw_index, rate_index] = 0.005
    native_covariance[rate_index, yaw_index] = 0.005

    destination = GaussianState(native_mean, native_covariance)

    actual = conversion(
        common_mean, common_covariance, destination
    )

    c = np.cos(native_mean[yaw_index])
    s = np.sin(native_mean[yaw_index])

    np.testing.assert_allclose(actual.x[:2], common_mean[:2])
    np.testing.assert_allclose(
        actual.x[2],
        common_mean[2] * c + common_mean[3] * s,
    )

    if include_acceleration:
        np.testing.assert_allclose(
            actual.x[3],
            common_mean[4] * c + common_mean[5] * s,
        )

    retained = [yaw_index, rate_index]

    np.testing.assert_allclose(
        actual.x[retained],
        destination.x[retained],
    )
    np.testing.assert_allclose(
        actual.P[np.ix_(retained, retained)],
        destination.P[np.ix_(retained, retained)],
    )

    assert_valid_covariance(actual.P)

    # Conversion must not modify its inputs.
    np.testing.assert_array_equal(destination.x, native_mean)
    np.testing.assert_array_equal(destination.P, native_covariance)

    name = "CTRA" if include_acceleration else "CTRV"
    print(f"PASS: {name} low-speed mean and retained uncertainty")


def check_adapter_routing():
    bank = build_imm()
    source = bank.filters[0]  # CV

    source_state = GaussianState(
        np.array([10., 20., 0., 0.]),
        np.diag([1., 1., 0.1, 0.1]),
    )

    for index, conversion in [
        (2, common_to_ctrv_low_speed),
        (3, common_to_ctra_low_speed),
    ]:
        destination = bank.filters[index]
        snapshot = destination.state_snapshot()

        common = source.to_common(
            source_state.x, source_state.P
        )
        expected = conversion(common.x, common.P, snapshot)

        actual = destination.convert_from(
            source,
            source_state,
            destination_state=snapshot,
        )

        np.testing.assert_allclose(actual.x, expected.x)
        np.testing.assert_allclose(actual.P, expected.P)

        # Low-speed routing requires an explicit destination snapshot.
        try:
            destination.convert_from(source, source_state)
        except ValueError:
            pass
        else:
            raise AssertionError("Missing destination snapshot accepted.")

    print("PASS: adapter low-speed routing")


def check_stopped_bank():
    imm = build_imm()

    # Install valid native stopped beliefs directly.
    # Heading and turn-rate uncertainty remain available.
    for adapter in imm.filters:
        state = adapter.state_snapshot()
        x = state.x.copy()

        if x.size == 4:           # CV
            x[2:4] = 0.
        elif adapter.state_angle_indices == (3,):  # CTRV
            x[2] = 0.
            x[4] = 0.
        elif adapter.state_angle_indices == (4,):  # CTRA
            x[2:4] = 0.
            x[5] = 0.
        else:                    # CA
            x[2:] = 0.

        adapter.replace_state(x, state.P)

    # First inspect interaction before running prediction.
    before = [a.state_snapshot() for a in imm.filters]

    predicted = imm.predict_mode_probabilities()
    mixed = imm.mix_states(imm.mixing_probabilities(predicted))

    for state in mixed:
        assert_valid_covariance(state.P)

    for adapter, original in zip(imm.filters, before):
        current = adapter.state_snapshot()
        np.testing.assert_array_equal(current.x, original.x)
        np.testing.assert_array_equal(current.P, original.P)

    # Several cycles at a stationary position.
    for _ in range(10):
        imm.predict(0.2)
        imm.update(np.array([0., 0.]))

        for adapter in imm.filters:
            assert_valid_covariance(adapter.state_snapshot().P)

        mean, covariance = imm.combine_estimates()
        assert np.all(np.isfinite(mean))
        assert_valid_covariance(covariance)

        np.testing.assert_allclose(
            imm.mode_probabilities.sum(), 1., atol=1e-12
        )

    print("PASS: stopped four-model interaction and filtering")


def main():
    check_fallback(common_to_ctrv_low_speed, False)
    check_fallback(common_to_ctra_low_speed, True)
    check_adapter_routing()
    check_stopped_bank()

    print("All low-speed checks passed.")


if __name__ == "__main__":
    main()