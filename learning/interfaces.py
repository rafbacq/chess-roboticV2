"""
Learning policy interface and policy registry.

Defines the abstract interface that ALL policies (learned or heuristic)
must implement, plus a registry for hot-swapping policies at runtime.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class LearningPolicy(ABC):
    """
    Abstract interface for manipulation policies (learned or heuristic).

    Every policy — whether a PPO-trained neural network or a simple
    heuristic — implements this interface. This allows the manipulation
    stack to be agnostic about whether a learned or classical policy
    is being used.
    """

    @abstractmethod
    def predict(self, observation: dict) -> dict:
        """
        Given an observation dict, return an action dict.

        The exact keys in observation and action depend on the subtask
        (grasp, placement, retry, etc.).
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this policy is loaded and ready to use."""
        ...

    @abstractmethod
    def get_fallback_policy_name(self) -> str:
        """Name of the heuristic fallback if this policy is unavailable."""
        ...

    @abstractmethod
    def get_name(self) -> str:
        """Unique name of this policy."""
        ...


class LearnedPolicy(LearningPolicy):
    """
    Wrapper for a learned policy loaded from a checkpoint.

    Loads a PyTorch or ONNX model and runs inference.
    Falls back to a named heuristic if the model fails to load.
    """

    def __init__(
        self,
        model_path: str,
        fallback_name: str,
        device: str = "cpu",
    ) -> None:
        self._model_path = Path(model_path)
        self._fallback_name = fallback_name
        self._device = device
        self._model = None
        self._loaded = False

        self._try_load()

    def _try_load(self) -> None:
        """Attempt to load the model. Fails gracefully."""
        if not self._model_path.exists():
            logger.warning(f"Model not found: {self._model_path}")
            return

        try:
            import torch
            self._model = torch.jit.load(str(self._model_path), map_location=self._device)
            self._model.eval()
            self._loaded = True
            logger.info(f"Loaded learned policy from {self._model_path}")
        except Exception as e:
            logger.warning(f"Failed to load learned policy: {e}")
            self._loaded = False

    def predict(self, observation: dict) -> dict:
        if not self._loaded or self._model is None:
            raise RuntimeError(
                f"Learned policy not loaded. Use fallback: {self._fallback_name}"
            )

        import torch

        # Convert observation to tensor
        obs_array = observation.get("obs_array", np.zeros(28))
        obs_tensor = torch.tensor(obs_array, dtype=torch.float32, device=self._device)
        obs_tensor = obs_tensor.unsqueeze(0)

        with torch.no_grad():
            action_tensor = self._model(obs_tensor)

        action = action_tensor.squeeze(0).cpu().numpy()
        return {"delta_ee": action[:6], "gripper_cmd": float(action[6])}

    def is_available(self) -> bool:
        return self._loaded

    def get_fallback_policy_name(self) -> str:
        return self._fallback_name

    def get_name(self) -> str:
        return f"learned:{self._model_path.stem}"


class PolicyRegistry:
    """
    Registry that maps subtask names to active policies.

    Supports dynamic switching between learned and heuristic policies.
    Always provides a working policy via fallback chain.

    Usage:
        registry = PolicyRegistry()
        registry.register("grasp", learned_policy)
        registry.register("grasp_fallback", heuristic_policy)

        policy = registry.get_policy("grasp")  # learned if available, else heuristic
    """

    def __init__(self) -> None:
        self._policies: dict[str, LearningPolicy] = {}
        self._fallbacks: dict[str, LearningPolicy] = {}

    def register(self, subtask: str, policy: LearningPolicy) -> None:
        """Register a policy for a subtask."""
        self._policies[subtask] = policy
        logger.info(f"Registered policy '{policy.get_name()}' for subtask '{subtask}'")

    def register_fallback(self, subtask: str, policy: LearningPolicy) -> None:
        """Register a fallback (heuristic) policy for a subtask."""
        self._fallbacks[subtask] = policy

    def get_policy(self, subtask: str) -> LearningPolicy:
        """
        Get the active policy for a subtask.

        Priority:
            1. Registered learned policy (if available)
            2. Registered fallback policy
            3. Raise KeyError

        Returns:
            The best available policy for the subtask.
        """
        # Try learned policy first
        if subtask in self._policies:
            policy = self._policies[subtask]
            if policy.is_available():
                return policy
            else:
                logger.info(
                    f"Learned policy for '{subtask}' unavailable, "
                    f"using fallback: {policy.get_fallback_policy_name()}"
                )

        # Try fallback
        if subtask in self._fallbacks:
            return self._fallbacks[subtask]

        # Try finding fallback by name from the learned policy
        if subtask in self._policies:
            fallback_name = self._policies[subtask].get_fallback_policy_name()
            if fallback_name in self._fallbacks:
                return self._fallbacks[fallback_name]

        available = list(self._policies.keys()) + list(self._fallbacks.keys())
        raise KeyError(
            f"No policy available for subtask '{subtask}'. "
            f"Available: {available}"
        )

    def list_subtasks(self) -> dict[str, str]:
        """List all subtasks and their active policy names."""
        result = {}
        for subtask in set(list(self._policies.keys()) + list(self._fallbacks.keys())):
            try:
                policy = self.get_policy(subtask)
                result[subtask] = policy.get_name()
            except KeyError:
                result[subtask] = "unavailable"
        return result
