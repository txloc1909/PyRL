from __future__ import annotations

from typing import Protocol
from typing import Tuple, Collection
from typing import Hashable, Callable


class State(Hashable):

    @property
    def is_terminal(self) -> bool:
        pass


class StateSpace(Collection[State]):

    def init_state(self) -> State:
        pass


class Action(Hashable):
    pass


class ActionSpace(Collection[Action]):
    pass


TransitionDynamic = Callable[[State, Action], float, State]
Policy            = Callable[[State], Action]


class Environment(Protocol):

    @property
    def current_state(self) -> State:
        pass

    @property
    def state_space(self) -> StateSpace:
        pass

    @property
    def action_space(self) -> ActionSpace:
        pass

    @property
    def _transition_func(self) -> TransitionDynamic:
        pass

    @property.getter
    def _transition_func(self, transition):
        pass

    def reset(self):
        pass

    def transition(self, action) -> Tuple[float, State]:
        return self._transition_func(self.current_state, action)
