"""Student implementation file for HW3 Part C.

Only edit this file unless course staff explicitly instruct otherwise.
"""

from __future__ import annotations

from typing import Any, Callable
import math
import random
from collections import Counter, defaultdict

from rescue_env import RescueEnv
from rescue_types import Action, State

AlphaSchedule = Callable[[int], float]
EpsilonSchedule = Callable[[int], float]


def estimate_transition_reward_model(
    trajectories: list[dict[str, Any]],
    actions: tuple[Action, ...],
    smoothing: float,
) -> dict[str, Any]:
    """Estimate P_hat and R_hat from offline transitions.

    The input trajectories list contains dictionaries with:
        state, action, reward, next_state, done

    Use Laplace smoothing for transition probabilities over the next states
    observed for each (state, action) pair. For each observed
    (state, action, next_state), use the mean observed reward for that triple.

    Return exactly this dictionary schema:
        {
            "transitions": {
                (state, action): [(probability, next_state, mean_reward), ...],
                ...
            },
            "actions_by_state": {
                state: (action1, action2, ...),
                ...
            },
            "states": (state1, state2, ...),
            "actions": actions,
        }

    Store actions in the same order as the actions argument. Store transition
    lists in a deterministic order, for example sorted by repr(next_state).
    States or actions with no observations should not invent transitions.
    """

    if smoothing < 0:
        raise ValueError("smoothing must be non-negative")

    transition_counts: dict[tuple[Any, Action], Counter[Any]] = defaultdict(Counter)
    reward_sums: dict[tuple[Any, Action, Any], float] = defaultdict(float)
    reward_counts: dict[tuple[Any, Action, Any], int] = defaultdict(int)
    observed_states: set[Any] = set()
    observed_actions_by_state: dict[Any, set[Action]] = defaultdict(set)

    for record in trajectories:
        state = record["state"]
        action = record["action"]
        reward = float(record["reward"])
        next_state = record["next_state"]

        observed_states.add(state)
        observed_states.add(next_state)
        observed_actions_by_state[state].add(action)

        transition_counts[(state, action)][next_state] += 1
        reward_sums[(state, action, next_state)] += reward
        reward_counts[(state, action, next_state)] += 1

    transitions: dict[tuple[Any, Action], list[tuple[float, Any, float]]] = {}

    # Iterating over sorted states and the supplied action order keeps the model
    # deterministic without inventing entries for unobserved state-action pairs.
    source_states = sorted(observed_actions_by_state, key=repr)
    for state in source_states:
        for action in actions:
            next_counter = transition_counts.get((state, action))
            if not next_counter:
                continue

            next_states = sorted(next_counter, key=repr)
            total_count = sum(next_counter.values())
            k = len(next_states)
            denominator = total_count + smoothing * k

            outcomes: list[tuple[float, Any, float]] = []
            for next_state in next_states:
                probability = (next_counter[next_state] + smoothing) / denominator
                triple = (state, action, next_state)
                mean_reward = reward_sums[triple] / reward_counts[triple]
                outcomes.append((probability, next_state, mean_reward))

            transitions[(state, action)] = outcomes

    actions_by_state: dict[Any, tuple[Action, ...]] = {}
    for state in source_states:
        observed_actions = observed_actions_by_state[state]
        actions_by_state[state] = tuple(
            action for action in actions if action in observed_actions
        )

    return {
        "transitions": transitions,
        "actions_by_state": actions_by_state,
        "states": tuple(sorted(observed_states, key=repr)),
        "actions": tuple(actions),
    }


def plan_with_estimated_model(
    model: dict[str, Any],
    start_state: State,
    horizon: int,
) -> dict[int, dict[State, Action]]:
    """Run finite-horizon planning on the learned model.

    Use the model schema returned by estimate_transition_reward_model. A state
    with no actions in actions_by_state should be treated as terminal with
    value 0. Break exact action-value ties according to the action order stored
    in model["actions"].
    """

    states = set(model.get("states", ()))
    states.add(start_state)

    states_by_time: dict[int, list[State]] = {
        t: [] for t in range(horizon + 1)
    }
    for state in sorted(states, key=repr):
        if isinstance(state, tuple) and len(state) >= 5:
            time_remaining = int(state[4])
            if 0 <= time_remaining <= horizon:
                states_by_time[time_remaining].append(state)

    values: dict[int, dict[State, float]] = {
        t: {} for t in range(horizon + 1)
    }
    policy: dict[int, dict[State, Action]] = {
        t: {} for t in range(1, horizon + 1)
    }

    for state in states_by_time[0]:
        values[0][state] = 0.0

    action_order: tuple[Action, ...] = tuple(model.get("actions", ()))
    actions_by_state = model.get("actions_by_state", {})
    transitions = model.get("transitions", {})

    for t in range(1, horizon + 1):
        for state in states_by_time[t]:
            observed_actions = set(actions_by_state.get(state, ()))
            legal_model_actions = tuple(
                action for action in action_order if action in observed_actions
            )

            # No observed action means that this state is terminal in the
            # estimated model, as required by the assignment.
            if not legal_model_actions:
                values[t][state] = 0.0
                continue

            best_action = legal_model_actions[0]
            best_value = _estimated_action_value(
                transitions,
                values[t - 1],
                state,
                best_action,
            )

            for action in legal_model_actions[1:]:
                value = _estimated_action_value(
                    transitions,
                    values[t - 1],
                    state,
                    action,
                )
                if _strictly_better(value, best_value):
                    best_value = value
                    best_action = action

            values[t][state] = best_value
            policy[t][state] = best_action

    return policy


def first_visit_mc(
    env: RescueEnv,
    policy: dict[int, dict[State, Action]] | dict[State, Action],
    n_episodes: int,
    gamma: float,
    seed: int,
) -> dict[State, float]:
    """Estimate V^pi from sampled episodes using first-visit Monte Carlo.

    Use random.Random(seed) to generate a fresh deterministic seed for each
    episode; do not pass the same seed to every rollout.
    """

    rng = random.Random(seed)
    return_sums: dict[State, float] = defaultdict(float)
    return_counts: dict[State, int] = defaultdict(int)
    horizon = _policy_horizon(policy, env.default_horizon)

    for _ in range(n_episodes):
        episode_seed = rng.randrange(2**32)
        result = env.simulate(policy, horizon=horizon, seed=episode_seed)
        trajectory = result["trajectory"]

        # Compute the return following every transition by scanning backward.
        returns_from_index = [0.0] * len(trajectory)
        episode_return = 0.0
        for i in range(len(trajectory) - 1, -1, -1):
            reward = float(trajectory[i][2])
            episode_return = reward + gamma * episode_return
            returns_from_index[i] = episode_return

        seen: set[State] = set()
        for i, transition in enumerate(trajectory):
            state = transition[0]
            if state in seen:
                continue
            seen.add(state)
            return_sums[state] += returns_from_index[i]
            return_counts[state] += 1

    return {
        state: return_sums[state] / return_counts[state]
        for state in return_sums
    }


def td_prediction(
    env: RescueEnv,
    policy: dict[int, dict[State, Action]] | dict[State, Action],
    n_episodes: int,
    alpha_schedule: AlphaSchedule,
    gamma: float,
    seed: int,
) -> dict[State, float]:
    """Estimate V^pi from online interaction using TD(0).

    Count TD updates from t=0 upward when calling alpha_schedule(t). Use a fresh
    deterministic episode seed derived from seed for each episode.
    """

    rng = random.Random(seed)
    values: dict[State, float] = {}
    update_index = 0
    horizon = _policy_horizon(policy, env.default_horizon)

    for _ in range(n_episodes):
        episode_seed = rng.randrange(2**32)
        state = env.reset(seed=episode_seed, horizon=horizon)

        while not env.is_terminal(state):
            action = _policy_action(env, policy, state)
            next_state, reward, done, _info = env.step(action)

            current_value = values.get(state, 0.0)
            next_value = 0.0 if done else values.get(next_state, 0.0)
            target = float(reward) + gamma * next_value
            alpha = alpha_schedule(update_index)

            values[state] = current_value + alpha * (target - current_value)
            update_index += 1
            state = next_state

    return values


def q_learning_rescue(
    env: RescueEnv,
    n_episodes: int,
    alpha_schedule: AlphaSchedule,
    epsilon_schedule: EpsilonSchedule,
    gamma: float,
    seed: int,
) -> tuple[dict[tuple[State, Action], float], dict[State, Action]]:
    """Learn a tabular Q policy with epsilon-greedy exploration.

    Count Q updates from t=0 upward when calling alpha_schedule(t) and
    epsilon_schedule(t). During exploration, choose uniformly from all legal
    env.actions(state). Break greedy ties using env.actions(state) order.
    """

    rng = random.Random(seed)
    q_values: dict[tuple[State, Action], float] = {}
    visited_states: set[State] = set()
    update_index = 0

    for _ in range(n_episodes):
        episode_seed = rng.randrange(2**32)
        state = env.reset(seed=episode_seed, horizon=env.default_horizon)

        while not env.is_terminal(state):
            legal_actions = env.actions(state)
            if not legal_actions:
                break

            visited_states.add(state)
            epsilon = epsilon_schedule(update_index)

            if rng.random() < epsilon:
                # The assignment defines exploration as uniform over all legal
                # actions, including the currently greedy action.
                action = rng.choice(legal_actions)
            else:
                action = _greedy_action_from_Q(q_values, state, legal_actions)

            next_state, reward, done, _info = env.step(action)
            _q_learning_update(
                env,
                q_values,
                state,
                action,
                next_state,
                float(reward),
                done,
                alpha_schedule(update_index),
                gamma,
            )

            update_index += 1
            state = next_state

    learned_policy = _greedy_policy_from_visited_states(
        env, q_values, visited_states
    )
    return q_values, learned_policy


def boltzmann_action(
    Q: dict[tuple[State, Action], float],
    state: State,
    actions: tuple[Action, ...],
    temperature: float,
) -> Action:
    """Sample an action according to softmax(Q(state, action) / temperature).

    Use the Python random module for sampling. Subtract the largest scaled
    Q-value before exponentiating for numerical stability.
    """

    if not actions:
        raise ValueError("actions must not be empty")
    if temperature <= 0:
        return _greedy_action_from_Q(Q, state, actions)

    scaled_values = [
        Q.get((state, action), 0.0) / temperature
        for action in actions
    ]
    largest = max(scaled_values)
    weights = [math.exp(value - largest) for value in scaled_values]

    threshold = random.random() * sum(weights)
    cumulative = 0.0
    for action, weight in zip(actions, weights):
        cumulative += weight
        if threshold < cumulative:
            return action

    # This can only be reached because of floating-point roundoff.
    return actions[-1]


def q_learning_boltzmann(
    env: RescueEnv,
    n_episodes: int,
    alpha_schedule: AlphaSchedule,
    temperature: float,
    gamma: float,
    seed: int,
) -> tuple[dict[tuple[State, Action], float], dict[State, Action]]:
    """Train Q-learning using Boltzmann exploration.

    This helper is not one of the required grader interfaces, but it makes the
    Part C.6 experiment reproducible using the same Q-learning update as
    q_learning_rescue.
    """

    q_values: dict[tuple[State, Action], float] = {}
    visited_states: set[State] = set()
    update_index = 0

    # boltzmann_action is required to use the module-level random generator.
    # Save and restore its state so this experiment is deterministic without
    # changing randomness elsewhere in the program.
    previous_random_state = random.getstate()
    random.seed(seed)
    try:
        for _ in range(n_episodes):
            episode_seed = random.randrange(2**32)
            state = env.reset(seed=episode_seed, horizon=env.default_horizon)

            while not env.is_terminal(state):
                legal_actions = env.actions(state)
                if not legal_actions:
                    break

                visited_states.add(state)
                action = boltzmann_action(
                    q_values,
                    state,
                    legal_actions,
                    temperature,
                )

                next_state, reward, done, _info = env.step(action)
                _q_learning_update(
                    env,
                    q_values,
                    state,
                    action,
                    next_state,
                    float(reward),
                    done,
                    alpha_schedule(update_index),
                    gamma,
                )

                update_index += 1
                state = next_state
    finally:
        random.setstate(previous_random_state)

    learned_policy = _greedy_policy_from_visited_states(
        env, q_values, visited_states
    )
    return q_values, learned_policy


def evaluate_rescue_agent(
    env: RescueEnv,
    policy: dict[int, dict[State, Action]] | dict[State, Action],
    n_episodes: int,
    seed: int,
) -> dict[str, Any]:
    """Return success rate, mean return, mean length, and failure breakdown.

    Use random.Random(seed) to generate a fresh deterministic seed for each
    evaluation episode. If a learned policy has no action for a reachable state,
    the provided environment uses its first legal action as a deterministic
    fallback.
    """

    rng = random.Random(seed)
    returns: list[float] = []
    lengths: list[int] = []
    terminal_reasons: Counter[str] = Counter()
    fallback_actions = 0
    horizon = _policy_horizon(policy, env.default_horizon)

    for _ in range(n_episodes):
        episode_seed = rng.randrange(2**32)
        result = env.simulate(policy, horizon=horizon, seed=episode_seed)

        returns.append(float(result["return"]))
        lengths.append(int(result["length"]))
        terminal_reasons[str(result["terminal_reason"])] += 1

        for state, _action, _reward, _next_state, _done in result["trajectory"]:
            if not _has_policy_action(policy, state):
                fallback_actions += 1

    success_count = terminal_reasons.get("success", 0)
    failure_breakdown = {
        reason: count
        for reason, count in sorted(terminal_reasons.items())
        if reason != "success"
    }

    return {
        "mean_return": sum(returns) / n_episodes if n_episodes else 0.0,
        "success_rate": success_count / n_episodes if n_episodes else 0.0,
        "mean_length": sum(lengths) / n_episodes if n_episodes else 0.0,
        "failure_breakdown": failure_breakdown,
        "returns": returns,
        "fallback_actions": fallback_actions,
    }


def _strictly_better(value: float, best_value: float) -> bool:
    """Compare values while ignoring only a few floating-point ULPs."""

    if not (math.isfinite(value) and math.isfinite(best_value)):
        return value > best_value
    tolerance = 4.0 * max(math.ulp(value), math.ulp(best_value))
    return value > best_value + tolerance


def _estimated_action_value(
    transitions: dict[tuple[State, Action], list[tuple[float, State, float]]],
    next_values: dict[State, float],
    state: State,
    action: Action,
) -> float:
    value = 0.0
    for probability, next_state, reward in transitions.get((state, action), []):
        value += probability * (reward + next_values.get(next_state, 0.0))
    return value


def _q_learning_update(
    env: RescueEnv,
    q_values: dict[tuple[State, Action], float],
    state: State,
    action: Action,
    next_state: State,
    reward: float,
    done: bool,
    alpha: float,
    gamma: float,
) -> None:
    current_q = q_values.get((state, action), 0.0)

    if done:
        next_best = 0.0
    else:
        next_actions = env.actions(next_state)
        next_best = max(
            (q_values.get((next_state, next_action), 0.0)
             for next_action in next_actions),
            default=0.0,
        )

    target = reward + gamma * next_best
    q_values[(state, action)] = current_q + alpha * (target - current_q)


def _greedy_policy_from_visited_states(
    env: RescueEnv,
    q_values: dict[tuple[State, Action], float],
    visited_states: set[State],
) -> dict[State, Action]:
    policy: dict[State, Action] = {}

    for state in sorted(visited_states, key=repr):
        if env.is_terminal(state):
            continue
        actions = env.actions(state)
        if actions:
            policy[state] = _greedy_action_from_Q(q_values, state, actions)

    return policy


def _policy_horizon(
    policy: dict[int, dict[State, Action]] | dict[State, Action],
    default: int,
) -> int:
    if policy and all(isinstance(key, int) for key in policy):
        return max(policy)
    return default


def _has_policy_action(
    policy: dict[int, dict[State, Action]] | dict[State, Action],
    state: State,
) -> bool:
    time_remaining = state[4]
    time_slice = policy.get(time_remaining)
    if isinstance(time_slice, dict):
        return state in time_slice
    return state in policy


def _policy_action(
    env: RescueEnv,
    policy: dict[int, dict[State, Action]] | dict[State, Action],
    state: State,
) -> Action:
    time_remaining = state[4]
    time_slice = policy.get(time_remaining)
    if isinstance(time_slice, dict):
        action = time_slice.get(state)
        if action is not None:
            return action
    else:
        action = policy.get(state)  # type: ignore[arg-type]
        if action is not None and not isinstance(action, dict):
            return action

    return env.actions(state)[0]


def _greedy_action_from_Q(
    Q: dict[tuple[State, Action], float],
    state: State,
    actions: tuple[Action, ...],
) -> Action:
    if not actions:
        raise ValueError("actions must not be empty")

    best_action = actions[0]
    best_value = Q.get((state, best_action), 0.0)

    for action in actions[1:]:
        value = Q.get((state, action), 0.0)
        if _strictly_better(value, best_value):
            best_value = value
            best_action = action

    return best_action


def default_alpha_schedule(t: int) -> float:
    """Alpha schedule required by the assignment."""

    return 0.5 / (1 + (t // 1000))


def constant_epsilon(_: int) -> float:
    return 0.1


def decaying_epsilon(t: int) -> float:
    return max(0.02, 1 / ((t + 1) ** 0.5))