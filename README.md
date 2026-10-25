# Rescue Planning & Reinforcement Learning

Planning and tabular reinforcement learning implementations for a rescue environment, developed as part of an Artificial Intelligence course at the Technion.

The project covers decision-making with a known transition model, planning from an estimated model, and learning through interaction.

## Implemented Methods

### Planning with a Known Model

- One-step expected action-value computation.
- Finite-horizon dynamic programming.
- Time-dependent policies.
- Policy rollouts and comparisons across time horizons.

### Learning an Environment Model

- Transition probability estimation from recorded experience.
- Mean reward estimation for observed transitions.
- Laplace smoothing over observed successor states.
- Finite-horizon planning using the estimated model.

### Policy Evaluation

- First-visit Monte Carlo prediction.
- TD(0) prediction.

### Control and Exploration

- Tabular Q-learning.
- Epsilon-greedy exploration.
- Constant and decaying epsilon schedules.
- Boltzmann exploration with numerically stabilized softmax sampling.

## Evaluation Utilities

The implementation collects:

- Success rate.
- Mean episode return.
- Mean episode length.
- Failure counts by terminal reason.
- Fallback action counts for incomplete policies.

Episode seeds are derived from a supplied seed to support reproducible experiments.

## Files

| File | Purpose |
|---|---|
| `planning_rescue.py` | Known-model planning and policy rollout utilities |
| `learning_rescue.py` | Model estimation, prediction, Q-learning, and evaluation |

## Requirements

- Python 3.10 or newer
- Course-provided `rescue_env.py`
- Course-provided `rescue_types.py`

The two implementation modules otherwise use the Python standard library.

## Usage

With a configured `RescueEnv` instance:

    from planning_rescue import finite_horizon_dp
    from learning_rescue import evaluate_rescue_agent

    values, policy = finite_horizon_dp(env, horizon=20)

    results = evaluate_rescue_agent(
        env,
        policy,
        n_episodes=100,
        seed=42,
    )

    print(results)

Here, `env` must be initialized using the course environment.

## Current Packaging Status

The uploaded submission contains the planning and learning modules but does not include `rescue_env.py` or `rescue_types.py`.

These environment files are required to execute the implementations. No benchmark results are claimed in this repository yet.

## Academic Context

Developed as a course assignment at the Technion. The planning and learning implementations integrate with a course-provided rescue environment.
