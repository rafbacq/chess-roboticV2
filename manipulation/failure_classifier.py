"""
Failure classification and recovery logic for chess manipulation.

Maps detected failure modes to appropriate recovery actions,
implementing a decision tree with configurable retry limits.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from chess_core.interfaces import (
    ChessMove,
    ExecutionStatus,
    FailureEvent,
    RecoveryAction,
)

logger = logging.getLogger(__name__)


@dataclass
class RecoveryConfig:
    """Configuration for failure recovery behavior."""
    max_retry_same_grasp: int = 2
    max_retry_regrasp: int = 2
    max_total_retries: int = 5
    slow_approach_velocity_scale: float = 0.1
    wider_clearance_factor: float = 1.5


class FailureClassifier:
    """
    Classifies execution failures and recommends recovery actions.

    Decision tree:
        PICKUP_FAILED → retry same grasp (up to N times) → reobserve → request human
        PIECE_SLIPPED → reobserve + regrasp → request human
        PLACE_OFFCENTER → retry with slower approach → request human
        COLLISION → replan with wider clearance → request human
        PLANNER_FAILED → replan with wider clearance → request human
        TIMEOUT → retry → request human
        BOARD_MISMATCH → reobserve → reconcile → request human
        CAMERA_LOW_CONFIDENCE → reobserve → request human
        CALIBRATION_DRIFT → recalibrate → request human
    """

    RECOVERY_MAP: dict[ExecutionStatus, list[RecoveryAction]] = {
        ExecutionStatus.PICKUP_FAILED: [
            RecoveryAction.RETRY_SAME_GRASP,
            RecoveryAction.REOBSERVE_AND_REGRASP,
            RecoveryAction.SLOW_APPROACH,
            RecoveryAction.REQUEST_HUMAN,
        ],
        ExecutionStatus.PIECE_SLIPPED: [
            RecoveryAction.REOBSERVE_AND_REGRASP,
            RecoveryAction.SLOW_APPROACH,
            RecoveryAction.REQUEST_HUMAN,
        ],
        ExecutionStatus.PLACE_OFFCENTER: [
            RecoveryAction.RETRY_SAME_GRASP,
            RecoveryAction.SLOW_APPROACH,
            RecoveryAction.REQUEST_HUMAN,
        ],
        ExecutionStatus.COLLISION: [
            RecoveryAction.REPLAN_WIDER_CLEARANCE,
            RecoveryAction.SLOW_APPROACH,
            RecoveryAction.REQUEST_HUMAN,
        ],
        ExecutionStatus.PLANNER_FAILED: [
            RecoveryAction.REPLAN_WIDER_CLEARANCE,
            RecoveryAction.REQUEST_HUMAN,
        ],
        ExecutionStatus.TIMEOUT: [
            RecoveryAction.RETRY_SAME_GRASP,
            RecoveryAction.REQUEST_HUMAN,
        ],
        ExecutionStatus.BOARD_MISMATCH: [
            RecoveryAction.REOBSERVE_AND_REGRASP,
            RecoveryAction.REQUEST_HUMAN,
        ],
        ExecutionStatus.CAMERA_LOW_CONFIDENCE: [
            RecoveryAction.REOBSERVE_AND_REGRASP,
            RecoveryAction.REQUEST_HUMAN,
        ],
        ExecutionStatus.CALIBRATION_DRIFT: [
            RecoveryAction.REQUEST_HUMAN,  # needs recalibration
        ],
        ExecutionStatus.GRIPPER_FAULT: [
            RecoveryAction.REQUEST_HUMAN,
        ],
        ExecutionStatus.EMERGENCY_STOP: [
            RecoveryAction.REQUEST_HUMAN,
        ],
        ExecutionStatus.ILLEGAL_MOVE: [
            RecoveryAction.ABORT_GAME,
        ],
    }

    def __init__(self, config: RecoveryConfig | None = None) -> None:
        self.config = config or RecoveryConfig()
        self._attempt_counts: dict[str, int] = {}

    def classify_and_recommend(
        self, failure: FailureEvent,
    ) -> RecoveryAction:
        """
        Given a failure event, recommend the next recovery action.

        Tracks retry counts and escalates through the recovery chain.

        Args:
            failure: The failure event.

        Returns:
            Recommended RecoveryAction.
        """
        move_key = failure.move.uci_string
        self._attempt_counts.setdefault(move_key, 0)
        self._attempt_counts[move_key] += 1
        attempt = self._attempt_counts[move_key]

        recovery_chain = self.RECOVERY_MAP.get(
            failure.status,
            [RecoveryAction.REQUEST_HUMAN],
        )

        # Check total retry limit
        if attempt > self.config.max_total_retries:
            logger.warning(
                f"Max total retries ({self.config.max_total_retries}) exceeded "
                f"for move {move_key}. Requesting human intervention."
            )
            return RecoveryAction.REQUEST_HUMAN

        # Select the appropriate recovery action based on attempt number
        idx = min(attempt - 1, len(recovery_chain) - 1)
        action = recovery_chain[idx]

        logger.info(
            f"Failure: {failure.status.name} on move {move_key} "
            f"(attempt {attempt}). Recovery: {action.name}"
        )

        return action

    def reset_for_move(self, uci_string: str) -> None:
        """Reset retry count for a specific move."""
        self._attempt_counts.pop(uci_string, None)

    def reset_all(self) -> None:
        """Reset all retry counts."""
        self._attempt_counts.clear()
