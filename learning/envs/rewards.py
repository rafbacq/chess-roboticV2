"""
Reward function library for RL manipulation environments.

Each reward function is a standalone callable that takes an environment
info dict and returns a scalar reward. This makes rewards composable,
testable, and easy to swap.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


class RewardComponent(ABC):
    """Base class for reward components."""

    @abstractmethod
    def compute(self, info: dict[str, Any]) -> float:
        """Compute this reward component given environment info."""
        ...

    @abstractmethod
    def get_name(self) -> str:
        """Return a unique name for logging."""
        ...


class SuccessReward(RewardComponent):
    """Binary reward for task completion."""

    def __init__(self, reward: float = 10.0) -> None:
        self._reward = reward

    def compute(self, info: dict[str, Any]) -> float:
        return self._reward if info.get("success", False) else 0.0

    def get_name(self) -> str:
        return "success"


class StepPenalty(RewardComponent):
    """Per-step penalty to encourage efficiency."""

    def __init__(self, penalty: float = -0.5) -> None:
        self._penalty = penalty

    def compute(self, info: dict[str, Any]) -> float:
        return self._penalty

    def get_name(self) -> str:
        return "step_penalty"


class DistanceShapingReward(RewardComponent):
    """Shaped reward based on distance decrease to target."""

    def __init__(self, scale: float = 1.0) -> None:
        self._scale = scale
        self._prev_distance: float | None = None

    def compute(self, info: dict[str, Any]) -> float:
        current_dist = info.get("distance_to_piece", 0.0)
        if self._prev_distance is None:
            self._prev_distance = current_dist
            return 0.0

        delta = self._prev_distance - current_dist
        self._prev_distance = current_dist
        return self._scale * delta

    def get_name(self) -> str:
        return "distance_shaping"

    def reset(self) -> None:
        self._prev_distance = None


class ContactReward(RewardComponent):
    """One-time reward for first contact with piece."""

    def __init__(self, reward: float = 2.0) -> None:
        self._reward = reward
        self._given = False

    def compute(self, info: dict[str, Any]) -> float:
        if not self._given and info.get("contacted", False):
            self._given = True
            return self._reward
        return 0.0

    def get_name(self) -> str:
        return "contact"

    def reset(self) -> None:
        self._given = False


class CollisionPenalty(RewardComponent):
    """Penalty for collision with board or neighboring pieces."""

    def __init__(self, penalty: float = -5.0) -> None:
        self._penalty = penalty

    def compute(self, info: dict[str, Any]) -> float:
        return self._penalty if info.get("collision", False) else 0.0

    def get_name(self) -> str:
        return "collision"


class KnockoverPenalty(RewardComponent):
    """Penalty for knocking over a piece."""

    def __init__(self, penalty: float = -10.0) -> None:
        self._penalty = penalty

    def compute(self, info: dict[str, Any]) -> float:
        return self._penalty if info.get("knocked_over", False) else 0.0

    def get_name(self) -> str:
        return "knockover"


@dataclass
class CompositeReward:
    """
    Compose multiple reward components into a single reward function.

    Usage:
        reward_fn = CompositeReward([
            SuccessReward(10.0),
            StepPenalty(-0.5),
            DistanceShapingReward(1.0),
            ContactReward(2.0),
            CollisionPenalty(-5.0),
        ])
        total = reward_fn.compute(info)
        breakdown = reward_fn.get_breakdown(info)
    """
    components: list[RewardComponent]

    def compute(self, info: dict[str, Any]) -> float:
        return sum(c.compute(info) for c in self.components)

    def get_breakdown(self, info: dict[str, Any]) -> dict[str, float]:
        return {c.get_name(): c.compute(info) for c in self.components}

    def reset(self) -> None:
        for c in self.components:
            if hasattr(c, "reset"):
                c.reset()
