"""Student implementation file for HW3 Part B.

Only edit this file unless course staff explicitly instruct otherwise.
"""

from __future__ import annotations

from collections import Counter
import math
import random
from typing import Any

from rescue_env import RescueEnv
from rescue_types import Action, State


def action_value(
    env: RescueEnv,
    V_next: dict[State, float],
    state: State,
    action: Action,
) -> float:
    """Return the one-step finite-horizon backup for one action.

    Formula:
        sum_{s'} P(s' | s,a) * (R(s,a,s') + V_next[s'])

    Missing states in V_next are treated as having value 0.0. This is useful
    for terminal successor states and for small test environments.
    """

    value = 0.0
    for probability, next_state, reward in env.transitions(state, action):
        value += probability * (reward + V_next.get(next_state, 0.0))
    return value


def finite_horizon_dp(
    env: RescueEnv,
    horizon: int,
) -> tuple[
    dict[int, dict[State, float]],
    dict[int, dict[State, Action]],
]:
    """Return V[t][s] and a time-dependent policy pi[t][s].

    The state already includes time_remaining. Reachable states are generated
    once and grouped by their embedded time value. Terminal states have value
    zero and do not receive policy entries.

    Action-value ties are broken by preserving the first action returned
    by env.actions(state). A tiny tolerance avoids treating floating-point
    roundoff as a real improvement.
    """

    reachable_states = env.reachable_states(horizon)

    states_by_time: dict[int, list[State]] = {
        t: [] for t in range(horizon + 1)
    }
    for state in reachable_states:
        time_remaining = state[4]
        if 0 <= time_remaining <= horizon:
            states_by_time[time_remaining].append(state)

    V: dict[int, dict[State, float]] = {
        t: {} for t in range(horizon + 1)
    }
    policy: dict[int, dict[State, Action]] = {
        t: {} for t in range(1, horizon + 1)
    }

    for state in states_by_time[0]:
        V[0][state] = 0.0

    for t in range(1, horizon + 1):
        for state in states_by_time[t]:
            if env.is_terminal(state):
                V[t][state] = 0.0
                continue

            actions = env.actions(state)
            if not actions:
                V[t][state] = 0.0
                continue

            # Starting with the first legal action and replacing it only for a
            # strictly larger value implements the required exact tie-breaking
            # rule: exact ties keep the earlier action from env.actions(state).
            best_action = actions[0]
            best_value = action_value(env, V[t - 1], state, best_action)

            for action in actions[1:]:
                value = action_value(env, V[t - 1], state, action)
                if _strictly_better(value, best_value):
                    best_value = value
                    best_action = action

            V[t][state] = best_value
            policy[t][state] = best_action

    return V, policy


def _strictly_better(value: float, best_value: float) -> bool:
    """Return True only for a numerically meaningful strict improvement.

    A few ULPs are ignored because mathematically equal Bellman sums can differ
    by one floating-point rounding step. This preserves deterministic tie
    breaking without hiding ordinary small improvements.
    """

    if not (math.isfinite(value) and math.isfinite(best_value)):
        return value > best_value
    tolerance = 4.0 * max(math.ulp(value), math.ulp(best_value))
    return value > best_value + tolerance


def rollout_time_dependent_policy(
    env: RescueEnv,
    policy: dict[int, dict[State, Action]],
    n_episodes: int,
    seed: int,
) -> dict[str, Any]:
    """Simulate a time-dependent policy and return summary statistics.

    A local random-number generator derives a different deterministic seed for
    each episode. The returned failure breakdown excludes successful episodes.
    """

    rng = random.Random(seed)
    returns: list[float] = []
    lengths: list[int] = []
    terminal_reasons: Counter[str] = Counter()

    horizon = max(policy) if policy else env.default_horizon

    for _ in range(n_episodes):
        episode_seed = rng.randrange(2**32)
        result = env.simulate(policy, horizon=horizon, seed=episode_seed)

        returns.append(float(result["return"]))
        lengths.append(int(result["length"]))
        terminal_reasons[str(result["terminal_reason"])] += 1

    success_count = terminal_reasons.get("success", 0)
    failure_breakdown = {
        reason: count
        for reason, count in sorted(terminal_reasons.items())
        if reason != "success"
    }

    if n_episodes == 0:
        mean_return = 0.0
        success_rate = 0.0
        mean_length = 0.0
    else:
        mean_return = sum(returns) / n_episodes
        success_rate = success_count / n_episodes
        mean_length = sum(lengths) / n_episodes

    return {
        "mean_return": mean_return,
        "success_rate": success_rate,
        "mean_length": mean_length,
        "failure_breakdown": failure_breakdown,
        "returns": returns,
    }


def compare_policy_slices(
    env: RescueEnv,
    policy: dict[int, dict[State, Action]],
    states_to_show: list[State],
) -> list[dict[str, Any]]:
    """Return a compact table of policy changes for selected states.

    For each supplied state, keep its position, carrying status, and battery
    fixed, then collect every available policy action across time. Rows with no
    matching policy entries are omitted.
    """

    # The environment argument is part of the required interface. The policy
    # already contains only legal, non-terminal entries, so no extra environment
    # query is needed here.
    _ = env

    rows: list[dict[str, Any]] = []
    for state in states_to_show:
        row, col, carrying, battery, _time_remaining = state
        actions_by_time: dict[int, Action] = {}

        for t in sorted(policy):
            candidate: State = (row, col, carrying, battery, t)
            action = policy[t].get(candidate)
            if action is not None:
                actions_by_time[t] = action

        if actions_by_time:
            rows.append({
                "state": state,
                "actions_by_time": actions_by_time,
            })

    return rows