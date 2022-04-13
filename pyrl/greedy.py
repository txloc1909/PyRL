from random import random, choice

from pyrl.core import State, Action
from pyrl.core import StateValueFunction
from pyrl.core import ActionValueFunction
from pyrl.core import Policy
from pyrl.modelfree.temporaldifference import QTable


class GreedyPolicy(Policy):

    def __init__(self, qtable: QTable):
        self._qtable = qtable

    def __call__(self, state: State) -> Action:
        return self._table.max_q_action(state)


class EpsilonGreedyPolicy(Policy):

    def __init__(self, qtable: QTable, epsilon: float):
        if epsilon < 0.0 or epsilon > 1.0:
            raise ValueError("Epsilon must be within [0, 1]")

        self._qtable = qtable
        self._epsilon = epsilon

    @property
    def epsilon(self):
        return self._epsilon

    def __call__(self, state: State) -> Action:
        if random() < self.epsilon:
            return choice(self._qtable.actions)
        else:
            return self._table.max_q_action(state)
