"""Single-object stop/restart and sensor-dropout integration scenario.

Run: python -m Perception.experiments.imm_stop_restart
No regime labels are supplied to the estimator. Truth is deterministic.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from Perception.experiments.imm_setup import build_imm
from Perception.tracking.single_target import SingleTargetTracker
from Perception.tracking.types import Measurement


def run(seed=2026, output=None, transition_matrix=None):
    imm = build_imm()
    if transition_matrix is not None:
        imm.set_transition_matrix(transition_matrix)
    wrapper = SingleTargetTracker(imm, coordinate_frame='world',
                                  initial_timestamp_s=0., sensor_id='position')
    rng = np.random.default_rng(seed)
    dt = .2
    position = np.zeros(2)
    speed, yaw = 8., 0.
    rows, truth, weights, missing, errors = [], [], [], [], []
    nodes, quadrature_weights = np.polynomial.legendre.leggauss(8)
    for k in range(1, 121):
        t0, t = (k - 1) * dt, k * dt
        acceleration = -2. if 4. <= t0 < 8. else 2. if 12. <= t0 < 16. else 0.
        omega = .2 if t0 >= 16. else 0.
        tau = .5 * dt * (nodes + 1.)
        heading = yaw + omega * tau
        velocity = (speed + acceleration * tau)[:, None] * np.column_stack(
            [np.cos(heading), np.sin(heading)])
        position += .5 * dt * (quadrature_weights @ velocity)
        speed = max(0., speed + acceleration * dt)
        yaw += omega * dt
        dropout = 9. <= t <= 10. or 18. <= t <= 19.
        measurement = None if dropout else Measurement(
            position + rng.normal(0., .7, 2), timestamp_s=t,
            coordinate_frame='world', sensor_id='position',
            measurement_space_identity=imm.measurement_space_identity)
        result = wrapper.process(measurement, timestamp_s=t)
        assert result.timestamp_s == t
        assert np.all(np.isfinite(result.x))
        eig = np.linalg.eigvalsh(result.P)
        assert eig.min() >= -1e-8 * max(1., abs(eig).max())
        np.testing.assert_allclose(imm.mode_probabilities.sum(), 1., atol=1e-12)
        for adapter in imm.filters:
            state = adapter.state_snapshot()
            eig = np.linalg.eigvalsh(state.P)
            assert eig.min() >= -1e-8 * max(1., abs(eig).max())
        rows.append(result.x)
        truth.append(np.r_[position, speed])
        weights.append(imm.mode_probabilities.copy())
        missing.append(dropout)
        errors.append(np.linalg.norm(result.x[:2] - position))
    rows, truth, weights = map(np.asarray, (rows, truth, weights))
    times = dt * np.arange(1, 121)
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    print(f'PASS: stop/restart + two 1-second dropout windows; position RMSE {rmse:.3f} m')
    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output / 'stop_restart.npz', times=times, truth=truth,
                            estimates=rows, probabilities=weights, dropout=missing)
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes[0, 0].plot(*truth[:, :2].T, 'k', label='Truth')
        axes[0, 0].plot(*rows[:, :2].T, label='IMM')
        axes[0, 0].set(xlabel='x [m]', ylabel='y [m]', title='Stop, restart, then turn')
        axes[0, 0].axis('equal')
        axes[0, 0].legend()
        axes[0, 1].plot(times, truth[:, 2], label='True speed')
        axes[0, 1].plot(times, np.linalg.norm(rows[:, 2:4], axis=1), label='IMM speed')
        axes[0, 1].set(ylabel='Speed [m/s]', title='Stop 8–12 s; restart 12–16 s')
        axes[0, 1].legend()
        axes[1, 0].plot(times, errors)
        axes[1, 0].set(ylabel='Position error [m]', title=f'Position RMSE {rmse:.3f} m')
        for j, name in enumerate(('CV', 'CA', 'CTRV', 'CTRA')):
            axes[1, 1].plot(times, weights[:, j], label=name)
        axes[1, 1].set(ylabel='Probability', ylim=(0, 1), title='Posterior mode probabilities')
        axes[1, 1].legend()
        for ax in (axes[0, 1], axes[1, 0], axes[1, 1]):
            ax.set_xlabel('Time [s]')
            for start, end in ((9, 10), (18, 19)):
                ax.axvspan(start, end, color='grey', alpha=.2)
            ax.grid(alpha=.2)
        fig.suptitle('Actual filters + IMM + SingleTargetTracker; grey spans: no observations')
        fig.tight_layout()
        fig.savefig(output / 'stop_restart.png', dpi=150)
        plt.close(fig)
    return rmse


if __name__ == '__main__':
    run(output=Path(__file__).resolve().parent / 'imm_integration_results')
