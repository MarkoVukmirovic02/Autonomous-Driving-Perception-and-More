"""Run from the project root: python -m Perception.experiments.imm_noise_comparison.

Uses build_imm() from imm_setup.py and the user's actual filter classes.
Truth is prescribed, deterministic switching motion; sensor noise is Gaussian.
Diffusion intensities match directly driven state variances at REFERENCE_DT,
not complete discrete Q matrices. Nonlinear diffusion Q is locally linearized.
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from Perception.experiments.imm_setup import build_imm
from Perception.tracking.imm import IMM
from Perception.tracking.imm_adapter import IMMFilterAdapter
from Perception.motion_models.linear.cv import ConstantVelocityModel
from Perception.motion_models.linear.ca import ConstantAccelerationModel
from Perception.motion_models.nonlinear.ctrv import CTRVModel
from Perception.motion_models.nonlinear.ctra import CTRAModel

REFERENCE_DT = 0.2
NAMES = ('KF-CV', 'KF-CA', 'EKF-CTRV', 'UKF-CTRA', 'IMM')
PHASES = ((0., 10., 'Constant velocity'), (10., 17., 'Turn'),
          (17., 23., 'Acceleration'), (23., 30., 'Accelerating turn'))
COLORS = ('tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple')


def motion_models(formulation):
    if formulation == 'interval':
        return [
            ConstantVelocityModel(2, acceleration_std=0.5),
            ConstantAccelerationModel(2, jerk_std=0.7),
            CTRVModel(acceleration_std=0.7, yaw_acceleration_std=0.15),
            CTRAModel(jerk_std=0.9, yaw_acceleration_std=0.15),
        ]
    # Match Var(delta v), Var(delta a), or Var(delta omega) at reference dt:
    # sigma^2 * reference_dt^2 = q * reference_dt.
    return [
        ConstantVelocityModel(2, noise_model='white_acceleration',
                              noise_intensity=0.5**2 * REFERENCE_DT),
        ConstantAccelerationModel(2, noise_model='white_jerk',
                                  noise_intensity=0.7**2 * REFERENCE_DT),
        CTRVModel(noise_model='continuous_diffusion',
                  speed_noise_intensity=0.7**2 * REFERENCE_DT,
                  yaw_rate_noise_intensity=0.15**2 * REFERENCE_DT),
        CTRAModel(noise_model='continuous_diffusion',
                  jerk_noise_intensity=0.9**2 * REFERENCE_DT,
                  yaw_rate_noise_intensity=0.15**2 * REFERENCE_DT),
    ]


def make_bank(formulation):
    # Reconstruct filters rather than changing motion_model after construction:
    # EKF/UKF may have already bound their dynamics and Jacobian callables.
    template = build_imm()
    adapters = []
    for prototype, motion in zip(template.filters, motion_models(formulation)):
        original = prototype.filter
        initial = original.state_snapshot()
        kwargs = dict(x0=initial.x, P0=initial.P, motion_model=motion,
                      measurement_model=original.measurement_model)
        if original.state_angle_indices:
            kwargs['state_angle_indices'] = original.state_angle_indices
        estimator = type(original)(**kwargs)
        adapters.append(IMMFilterAdapter(
            estimator, to_common=prototype.to_common,
            to_native=prototype.to_native, state_space=prototype.state_space,
            low_speed_conversion=prototype.low_speed_conversion,
            minimum_speed=prototype.minimum_speed,
            common_space=prototype.common_space,
            coordinate_frame=prototype.coordinate_frame,
            measurement_frame=prototype.measurement_frame))
    return IMM(adapters, template.mode_probabilities, template.transition_matrix)


def trajectory(dt):
    steps = int(round(30. / dt))
    if not np.isclose(steps * dt, 30.):
        raise ValueError('dt must divide 30 seconds exactly')
    times = np.arange(steps + 1) * dt
    truth = np.zeros((steps + 1, 6))
    position = np.zeros(2)
    speed, yaw = 8., 0.
    truth[0] = [0., 0., 8., 0., 0., 0.]
    nodes, weights = np.polynomial.legendre.leggauss(8)
    # Independent quadrature of the physical velocity, not a filter model.
    for k in range(steps):
        left, end = times[k], times[k + 1]
        while left < end - 1e-12:
            phase = next(p for p in PHASES if left < p[1] - 1e-12)
            stop = min(end, phase[1])
            h = stop - left
            acceleration, omega = {
                'Constant velocity': (0., 0.), 'Turn': (0., 0.30),
                'Acceleration': (1.3, 0.), 'Accelerating turn': (1., -0.24),
            }[phase[2]]
            tau = 0.5 * h * (nodes + 1.)
            headings = yaw + omega * tau
            velocities = (speed + acceleration * tau)[:, None] * np.column_stack(
                [np.cos(headings), np.sin(headings)])
            position += 0.5 * h * (weights @ velocities)
            speed += acceleration * h
            yaw += omega * h
            left = stop
        tangent = np.array([np.cos(yaw), np.sin(yaw)])
        normal = np.array([-np.sin(yaw), np.cos(yaw)])
        truth[k + 1] = np.r_[position, speed * tangent,
                              acceleration * tangent + speed * omega * normal]
    return times, truth


def validate_belief(state):
    if not np.all(np.isfinite(state.x)) or not np.all(np.isfinite(state.P)):
        raise AssertionError('Nonfinite state or covariance')
    np.testing.assert_allclose(state.P, state.P.T, atol=1e-9, rtol=0)
    eig = np.linalg.eigvalsh(state.P)
    if eig.min() < -1e-8 * max(1., np.max(np.abs(eig))):
        raise AssertionError('Covariance is not positive semidefinite')


def run(formulation, times, truth, observations):
    imm = make_bank(formulation)
    # A separate bank provides independently running single-model baselines.
    single = make_bank(formulation).filters
    predictions, posteriors, probabilities = [], [], []
    for k in range(1, len(times)):
        dt = times[k] - times[k - 1]
        for adapter in single:
            adapter.predict(dt)
        imm.predict(dt)
        predictions.append(np.stack(
            [a.common_state().x for a in single] + [imm.combine_estimates()[0]]))
        for adapter in single:
            adapter.update(observations[k])
        imm.update(observations[k])
        posteriors.append(np.stack(
            [a.common_state().x for a in single] + [imm.combine_estimates()[0]]))
        for adapter in list(single) + list(imm.filters):
            validate_belief(adapter.state_snapshot())
        combined_x, combined_P = imm.combine_estimates()
        from Perception.tracking.types import GaussianState
        validate_belief(GaussianState(combined_x, combined_P))
        if np.any(imm.mode_probabilities < 0):
            raise AssertionError('Negative probability')
        np.testing.assert_allclose(imm.mode_probabilities.sum(), 1., atol=1e-12)
        probabilities.append(imm.mode_probabilities.copy())
    return dict(predicted=np.array(predictions), updated=np.array(posteriors),
                probabilities=np.array(probabilities))


def plot_result(formulation, times, truth, observations, result, output):
    t = times[1:]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax = axes[0, 0]
    ax.plot(truth[:, 0], truth[:, 1], 'k', lw=2, label='Truth')
    ax.scatter(observations[1:, 0], observations[1:, 1], s=4, c='0.8', label='Observations')
    for j, name in enumerate(NAMES):
        ax.plot(result['updated'][:, j, 0], result['updated'][:, j, 1],
                color=COLORS[j], lw=2 if j == 4 else 1, label=name)
    ax.set(xlabel='x [m]', ylabel='y [m]', title='Updated trajectories')
    ax.axis('equal')
    ax.legend(fontsize=8)
    for ax, key, title in [(axes[0, 1], 'predicted', 'Position error BEFORE observation'),
                           (axes[1, 0], 'updated', 'Position error AFTER observation')]:
        errors = np.linalg.norm(result[key][:, :, :2] - truth[1:, None, :2], axis=2)
        for j, name in enumerate(NAMES):
            ax.plot(t, errors[:, j], color=COLORS[j], label=name,
                    lw=2 if j == 4 else 1)
        ax.set(xlabel='Time [s]', ylabel='Error [m]', title=title)
        ax.legend(fontsize=8)
    ax = axes[1, 1]
    for j, name in enumerate(NAMES[:4]):
        ax.plot(t, result['probabilities'][:, j], color=COLORS[j], label=name)
    ax.set(xlabel='Time [s]', ylabel='Probability', ylim=(0, 1), title='IMM posterior mode probabilities')
    ax.legend(fontsize=8)
    for ax in [axes[0, 1], axes[1, 0], axes[1, 1]]:
        for start, end, label in PHASES:
            ax.axvspan(start, end, alpha=0.045, color=COLORS[list(PHASES).index((start, end, label))])
        for boundary in (10, 17, 23):
            ax.axvline(boundary, color='0.5', lw=0.8)
        ax.grid(alpha=0.2)
    fig.suptitle(f'{formulation}: actual KF/EKF/UKF + native CV/CA/CTRV/CTRA\n'
                 'CV 0–10 s | turn 10–17 s | acceleration 17–23 s | accelerating turn 23–30 s')
    fig.tight_layout()
    fig.savefig(output / f'{formulation}.png', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dt', type=float, default=0.2)
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).resolve().parent / 'imm_noise_results')
    args = parser.parse_args()
    if args.runs < 1 or not np.isfinite(args.dt) or args.dt <= 0:
        parser.error('runs and dt must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    times, truth = trajectory(args.dt)
    rows = []
    accumulators = {f: [] for f in ('interval', 'diffusion')}
    for run_index in range(args.runs):
        rng = np.random.default_rng(args.seed + run_index)
        observations = truth[:, :2] + rng.normal(0., 0.7, size=(len(times), 2))
        # The initialized belief is at t=0; observation[0] is not assimilated.
        for formulation in accumulators:
            result = run(formulation, times, truth, observations)
            accumulators[formulation].append(result)
            if run_index == 0:
                plot_result(formulation, times, truth, observations, result, args.output)
            np.savez_compressed(args.output / f'{formulation}_run_{run_index}.npz',
                                times=times, truth=truth, observations=observations, **result)
    for formulation, results in accumulators.items():
        print(f'\n{formulation.upper()} — {args.runs} observation-noise runs')
        for start, end, label in PHASES:
            # Classify by the dynamics used for the transition ending at t.
            # Exclude the first five seconds of initialization from CV metrics.
            mask = (times[1:] > max(start, 5.)) & (times[1:] <= end + 1e-10)
            if not mask.any():
                continue
            print(f'  {label}')
            for j, name in enumerate(NAMES):
                metrics = {}
                for key in ('predicted', 'updated'):
                    squared_errors = np.stack([
                        np.sum((r[key][mask, j, :2] - truth[1:][mask, :2])**2, axis=1)
                        for r in results])
                    metrics[key] = np.sqrt(np.mean(squared_errors))
                rows.append(dict(formulation=formulation, phase=label, estimator=name,
                                 prediction_rmse_m=metrics['predicted'],
                                 update_rmse_m=metrics['updated']))
                print(f"    {name:10s} prediction {metrics['predicted']:.3f} m; update {metrics['updated']:.3f} m")
    with (args.output / 'metrics.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'\nOutputs: {args.output.resolve()}')
    print('Pooled position RMSE across time and noise runs; deterministic truth.')
    print('Diffusion matches driven-state variance at reference dt=0.2 s, not full Q.')


if __name__ == '__main__':
    main()
