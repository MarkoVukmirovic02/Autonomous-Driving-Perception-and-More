import numpy as np

from Perception.experiments.imm_setup import build_imm
from Perception.tracking.imm import IMM


def assert_unchanged(imm, snapshots, probabilities):
    for adapter, original in zip(imm.filters, snapshots):
        current = adapter.state_snapshot()
        np.testing.assert_array_equal(current.x, original.x)
        np.testing.assert_array_equal(current.P, original.P)

    np.testing.assert_array_equal(
        imm.mode_probabilities, probabilities
    )


def check_single_mode():
    z = np.array([1.8, -0.3])
    R = np.diag([0.6, 0.8])

    for index, name in enumerate([
        "KF-CV", "KF-CA", "EKF-CTRV", "UKF-CTRA"
    ]):
        adapter = build_imm().filters[index]

        imm = IMM(
            filters=[adapter],
            mode_probabilities=[1.0],
            transition_matrix=[[1.0]],
        )
        imm.predict(0.2)

        snapshots = [adapter.state_snapshot()]
        probabilities = imm.mode_probabilities.copy()

        expected_prediction = adapter.measurement_prediction(R=R)
        actual_prediction = imm.measurement_prediction(R=R)

        np.testing.assert_allclose(
            actual_prediction.mean, expected_prediction.mean
        )
        np.testing.assert_allclose(
            actual_prediction.covariance,
            expected_prediction.covariance,
        )

        expected = adapter.innovation_diagnostics(z, R=R)
        actual = imm.innovation_diagnostics(z, R=R)

        for field in [
            "residual",
            "covariance",
            "nis",
            "log_likelihood",
            "whitened_residual",
        ]:
            np.testing.assert_allclose(
                getattr(actual, field),
                getattr(expected, field),
                rtol=1e-10,
                atol=1e-10,
            )

        assert_unchanged(imm, snapshots, probabilities)
        print(f"PASS: {name} measurement diagnostics equivalence")


def check_known_mixture():
    # Two independent actual CV filters.
    left = build_imm().filters[0]
    right = build_imm().filters[0]

    # Position covariance 0.5 I plus measurement noise 0.5 I
    # gives each component observation covariance I.
    P = np.diag([0.5, 0.5, 1.0, 1.0])
    left.replace_state(np.array([-2., 0., 8., 0.]), P)
    right.replace_state(np.array([2., 0., 8., 0.]), P)

    imm = IMM(
        filters=[left, right],
        mode_probabilities=[0.25, 0.75],
        transition_matrix=np.eye(2),
    )

    R = 0.5 * np.eye(2)
    z = np.zeros(2)

    snapshots = [a.state_snapshot() for a in imm.filters]
    probabilities = imm.mode_probabilities.copy()

    prediction = imm.measurement_prediction(R=R)
    diagnostics = imm.innovation_diagnostics(z, R=R)

    np.testing.assert_allclose(prediction.mean, [1., 0.])
    np.testing.assert_allclose(
        prediction.covariance, np.diag([4., 1.])
    )

    np.testing.assert_allclose(diagnostics.residual, [-1., 0.])
    np.testing.assert_allclose(diagnostics.nis, 0.25)

    expected_log_likelihood = -np.log(2 * np.pi) - 2.0

    np.testing.assert_allclose(
        diagnostics.log_likelihood, expected_log_likelihood
    )

    # Ensure likelihood is not taken from the moment-matched Gaussian.
    gaussian_log_likelihood = (
        -np.log(2 * np.pi)
        - 0.5 * np.log(4.0)
        - 0.5 * diagnostics.nis
    )

    assert not np.isclose(
        diagnostics.log_likelihood,
        gaussian_log_likelihood,
    )

    assert_unchanged(imm, snapshots, probabilities)

    print("PASS: known mixture mean and covariance")
    print("PASS: gating distance and component-mixture likelihood")
    print("PASS: diagnostics do not mutate the belief")

    # A zero-weight component must not influence the result.
    imm.mode_probabilities = np.array([1., 0.])
    prediction = imm.measurement_prediction(R=R)
    diagnostics = imm.innovation_diagnostics(z, R=R)
    reference = left.innovation_diagnostics(z, R=R)

    np.testing.assert_allclose(prediction.mean, [-2., 0.])
    np.testing.assert_allclose(prediction.covariance, np.eye(2))
    np.testing.assert_allclose(
        diagnostics.log_likelihood,
        reference.log_likelihood,
    )

    print("PASS: zero-weight component exclusion")


def main():
    check_single_mode()
    check_known_mixture()
    print("All measurement-mixture checks passed.")


if __name__ == "__main__":
    main()