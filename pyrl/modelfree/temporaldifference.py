from __future__ import annotations

from random import random
from typing import MutableMapping, DefaultDict
from collections import defaultdict

from pyrl.core import State, StateSpace, Action, ActionSpace
from pyrl.core import Policy
from pyrl.core import Environment
from pyrl.greedy import GreedyPolicy, EpsilonGreedyPolicy


class QTable(MutableMapping[Tuple[State, Action], float]):

    @property
    def states(self) -> Iterable[State]:
        """Return an iterable object, iterate through all possible states"""
        return NotImplemented

    @property
    def actions(self) -> Iterable[Action]:
        """Return an iterable object, iterate through all possible actions"""
        return NotImplemented

    def max_q_value(self, s: State) -> float:
        """Return the maximum q-value corresponding to a state"""
        return NotImplemented

    def max_q_action(self, s: State) -> Action:
        """Return the action with maximum q-value corresponding to a state"""
        return NotImplemented


class DefaultQTable(QTable):
    """Default QTable implementation.
    Using collections.defaultdict as the internal datastructure
    """

    @staticmethod
    def _default_value(s: State, a: Action) -> float:
        return 0.0 if s.is_terminal else random()

    def __init__(self, state_space: StateSpace, action_space: ActionSpace):
        self._state_space = state_space
        self._action_space = action_space
        self._table = defaultdict(default_factory=_default_value)

    def __getitem__(self, state_action) -> float:
        return self._table[state_action]

    def __setitem__(self, state_action, value):
        self._table[state_action] = value

    def __len__(self):
        return len(self.states) * len(self.actions)

    def __iter__(self):
        return iter(self._table)

    @property
    def states(self) -> Iterable[State]:
        return iter(self._state_space)

    @property
    def actions(self) -> Iterable[Action]:
        return iter(self._action_space)

    def max_q_value(self, state: State) -> float:
        return max(self._table[state, a] for a in self.actions)

    def max_q_action(self, state: State) -> Action:
        return max(self.actions, key=lambda a: self._table[state, a])


class QLearning:

    def __init__(self, env: Environment,
                 qtable_type: type[QTable] = DefaultQTable):
        self._env = env
        self.qtable_type = qtable_type

    def learn(self, num_episodes: int, alpha: float, gamma: float,
              epsilon: float, max_step: int = None) -> Policy:
        if max_step is None:
            max_step = float("inf")

        qtable = self.qtable_type(states_space=self._env.states_space,
                                  actions_space=self._env.actions_space)

        behavior_policy = EpsiloneGreedyPolicy(qtable=qtable, epsilon=epsilon)
        learning_policy = GreedyPolicy(qtable=qtable)

        for _ in range(num_episodes):
            env.reset()
            while not env.current_state.is_terminal and timestep < max_step:
                s = env.current_state
                a = behavior_policy(s)
                reward, new_s = env.transition(a)
                alt_a = learning_policy(new_s)
                qtable[s, a] = (1.0 - alpha) * qtable[s, a] +\
                        alpha * (reward + gamma * qtable[new_s, alt_a])

        return learning_policy
