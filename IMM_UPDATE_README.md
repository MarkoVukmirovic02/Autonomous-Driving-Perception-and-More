# Reusable single-object IMM update

This update finishes the current single-object heterogeneous IMM component.
It delegates filtering to the uploaded KF/EKF/UKF, mixes in destination-native
coordinates, and produces a Cartesian Gaussian output summary. MTT is not
required. This is an integration-ready component, not a tested CARLA application.

## Install

1. Back up your current project or commit it in git.
2. Extract `imm_generalized_update.zip` into the directory containing `Perception`,
   allowing replacement of the listed files. This archive is an update: it needs
   the other files from your supplied `imm_current.zip` / existing project.
3. Run from that project root:

```bash
python3 -m Perception.experiments.run_imm_checks
python3 -m Perception.experiments.imm_noise_comparison --runs 3
python3 -m Perception.experiments.imm_maneuver_visualization radar_imm_results
```

The first command runs the original five check groups, 17 additional integration
checks, and the stop/restart/dropout scenario. NumPy and SciPy are used by the
filters/models; Matplotlib is used for the examples. No new filtering library
is introduced. Do not execute tracking files directly: use package imports / `-m`.

## What changed

- `tracking/imm.py`: common output and predictive observation moments; component
  mixture likelihood; measurement container updates; explicit metadata checks;
  native bank checkpoints; invalid-dt preflight; zero-dt no-op; fail-stop state
  after a mutating failure; replacement of a known transition matrix.
- `tracking/imm_adapter.py`: explicit common coordinate identity, state and
  measurement frames, model-declared observation context. Existing four factory
  functions still delegate to raw filters and existing conversions.
- `tracking/single_target.py`: the estimator protocol now requires operations,
  not a single owned motion model. Gaussian replacement is an optional separate
  capability. Added wrapper checkpoints restoring the bank AND clock together.
- `tracking/types.py`: documented that IMM gating NIS and mixture likelihood
  describe different approximations. Existing containers remain compatible.
- `measurement_models/nonlinear/radar.py`: exposes a read-only comparable
  `measurement_context` key so modes cannot silently use different radar geometry.
  No radar projection or filter equations were changed.
- `tracking/__init__.py`: exports current symbols and removes stale nonexistent
  `IMMEstimate` / `IMMModeBank` export names.
- `experiments/imm_setup.py`: supplies the example's known `world` state frame.
- `experiments/imm_noise_comparison.py`: reconstructed banks now retain low-speed
  policy and coordinate metadata.
- `experiments/imm_maneuver_visualization.py`: migrated the older radar demo to the
  current IMM/adapters and current keyword-based noise constructors; initialization
  observation is not assimilated twice. Legacy removed class names are not aliased
  to an incompatible API.
- New integration checks, stop/restart experiment and one-command check runner.

All uploaded KF/EKF/UKF implementation files, motion-model files, and the previously
verified conversion formulas are unchanged. The source change manifest lists every
modified/new source with SHA-256 hashes against the uploaded archive.

## Construct and use

`Perception.experiments.imm_setup.build_imm()` is a complete position-observation
example. For an application, construct independent filters with compatible
observations, then wrap them:

```python
from Perception.tracking import IMM, SingleTargetTracker, Measurement
from Perception.tracking.imm_adapter import make_cv_adapter

# cv_filter already owns its motion/measurement models and initial belief.
cv_adapter = make_cv_adapter(
    cv_filter,
    acceleration_covariance=acceleration_prior_covariance,
    coordinate_frame="world",
    measurement_frame="world",  # position observation example
)
# Construct the other adapters similarly, then:
imm = IMM(adapters, mode_probabilities, transition_matrix)
tracker = SingleTargetTracker(
    imm,
    coordinate_frame="world",
    measurement_frame="world",
    initial_timestamp_s=initial_time,
    sensor_id="position",
)
measurement = Measurement(
    value=z, covariance=R, timestamp_s=capture_time,
    coordinate_frame="world", sensor_id="position",
    measurement_space_identity=imm.measurement_space_identity,
)
estimate = tracker.process(measurement)
```

For a missing observation:

```python
estimate = tracker.process(None, timestamp_s=next_capture_time)
```

This performs prediction only. It does not model information in the detection
absence (P_D, occlusion or field of view).

For diagnostics before association at a known time:

```python
tracker.predict(timestamp_s=capture_time)
diagnostics = tracker.innovation_diagnostics(measurement)
estimate = tracker.update(measurement)  # do not predict a second time
```

Raw `imm.predict(dt)` / `imm.update(z, R)` are supported for manually coordinated
callers. `imm.update_measurement(measurement)` validates observation identity/frame;
absolute time and sensor provenance belong to the wrapper.

## Mathematical contract

- Pi[i,j] = P(m_k=j | m_(k-1)=i); rows sum to one.
- mu_bar = Pi.T @ mu; mixing weight w[i,j] = mu[i] Pi[i,j] / mu_bar[j].
- Each source is converted to destination j before native mean/covariance mixing.
  Self/compatible native contributions bypass conversion. Covariance includes
  both within-mode uncertainty and between-mode disagreement.
- Native headings use circular means and wrapped residuals. Ambiguous circular
  means raise an error rather than inventing a confident heading.
- posterior mu_j is proportional to exp(component_log_likelihood_j) * mu_bar_j,
  normalized in log space. Likelihood is evaluated before the filter update.
- Output moments use each filter's common-state representation. They are output
  summaries; they are not copied back into every mode after each update.
- Predictive observation moments include component S_j (already containing R)
  and between-model differences. Measurement noise is not added a second time.
- Gating uses moment-matched NIS. Log-likelihood uses log(sum_j mu_j exp(ell_j)).
  The mixture density is not the density of the moment-matched Gaussian.

The supplied factories use native states:

- CV: [px,py,vx,vy]
- CA: [px,py,vx,vy,ax,ay]
- CTRV: [px,py,v,yaw,yaw_rate]
- CTRA: [px,py,v,a,yaw,yaw_rate]

Their common space is [px_m,py_m,vx_m_s,vy_m_s,ax_m_s2,ay_m_s2]. The IMM core
requires one Euclidean common output; another task (e.g. image boxes) needs
appropriate adapters/common semantics, not the planar conversion functions.

## Time and process-noise contract

A positive-dt predict applies ONE mode transition. Pi is a discrete per-step matrix,
not a switching rate per second. `predict(0)` is a no-op. For variable dt, supply an
appropriate Pi_k via `set_transition_matrix` before prediction if the mode process
must be consistent in physical time (e.g. from a mode generator).

The interval and diffusion examples use distinct Q formulations. Diffusion intensity
q=sigma^2*0.2 matches increment variance in the directly driven native quantity at
reference dt=0.2 s; the full covariance matrices are not equivalent. Nonlinear
continuous-diffusion Q uses the uploaded models' local linearization/discretization.
Truth in these experiments is prescribed deterministic motion, not an SDE sample.

## Failure and recovery

Invalid dt, observation identity/frame, and conversions are checked before bank
mutation where possible. A failure while replacing/predicting/updating a mode may
leave some modes changed. The IMM then sets `failed=True` and blocks further belief
use. It never claims to roll back arbitrary diagnostic caches or external effects.

Use a checkpoint from the SAME bank and unchanged configuration:

```python
checkpoint = tracker.checkpoint()
try:
    estimate = tracker.process(measurement)
except Exception:
    tracker.restore_checkpoint(checkpoint)
    raise  # inspect/correct the cause before retrying
```

The wrapper checkpoint restores both the estimator and its time. Calling only
`imm.restore_checkpoint()` does not restore a surrounding wrapper's time.
Checkpoints retain native means, covariances and mode probabilities; filter public
`replace_state` clears diagnostics. They are in-memory recovery objects, not a
portable serialization format. Mode/filter identities must match. The transition
matrix and declared geometry/configuration must match. Do not change models,
calibration or mutable provider state between capture/restore. Extra hidden state
in future custom estimators needs an extended checkpoint contract. If restoration
itself fails, rebuild the bank.

## Frames, sensors and CARLA boundary

Supply explicit frames in every factory for application use. Legacy `None` frames
are allowed only when consistently unspecified across the bank; they are a caller
assumption, not evidence that frames have been verified. Equal coordinate counts
never establish equal physical meanings. Adapter identities include ordering/units;
frame names do not perform transformations.

All modes must predict the SAME observation coordinates for one update. Radar
geometry contexts must match. For a moving ego sensor, update every mode's radar
`set_sensor_context(...)` to the capture-time pose/velocity before prediction/
diagnostics/update. The wrapper validates capture times, frames and optional sensor
ID, but does not synchronize devices, transform points, reorder delayed observations
or change observation models automatically.

Raw CARLA radar returns need preprocessing/object measurements and known association.
CARLA azimuth/range/radial-velocity sign, frame convention, ego-motion, and calibration
must be adapted explicitly. Camera boxes are not Cartesian positions. KF requires a
linear observation model; use EKF/UKF with the native radar models. The component
supports these models, but this update has not been executed inside CARLA.

## Approximation limits and scope

- Conversion covariances are first-order approximations; broad heading/speed
  distributions and strong nonlinearities need additional assessment.
- Low-speed cross-conversion retains the destination heading/turn-rate marginal,
  projects source velocity/acceleration and assumes independence of those inputs.
  It loses transverse information and is not an exact inverse. At exactly zero
  mean speed, first-order covariance propagation misses some higher-order terms.
- Native/common/native parameter equality is not guaranteed for negative speed;
  direct mean maps use a nonnegative-speed convention away from the fallback.
- The fallback needs a prior heading belief. Initializing a stopped target needs
  an explicit heading/turn-rate prior; do not infer confident yaw from zero velocity.
- Moment-matched gating on a mixture does not have exact Gaussian chi-square
  coverage. Angular component likelihoods inherit the filters' local wrapped-
  residual Gaussian approximation, not a globally normalized wrapped-density model.
- Numerical checks demonstrate specified behavior, not universal tracking accuracy
  or statistical covariance calibration. Stop/restart/dropout errors are retained
  in the example plots; Q, Pi and conversion priors remain tunable assumptions.
- No new MTT lifecycle, detector, association algorithm, sensor fusion, localization,
  safety controller or CARLA live sensor adapter is implemented in this update.

## Verification performed

The original five dedicated check modules pass unchanged. Seventeen new unittest
cases cover wrapper/protocol compatibility, detached snapshots/checkpoints, metadata
rejection, invalid inputs before mutation, zero dt, missing observations, transition
replacement, duplicate modes, raw-filter wrapper compatibility, injected failures,
clock+bank recovery, actual radar angle wrapping on both sides of +/-pi, radar
context rejection, and the association diagnostics interface without running MTT.

The stop/restart/dropout integration scenario completed with finite valid native
and output covariances, normalized probabilities, and position RMSE 1.130 m for
seed 2026. This is a single illustrative run, not a quality threshold.

The three-run interval/diffusion experiment reproduced the supplied result levels:
IMM prediction RMSE approximately 0.453 / 0.838 / 0.781 / 0.953 m across the four
regimes, with negligible formulation differences at the matched 0.2 s interval.
The migrated radar maneuver example also executes successfully.
