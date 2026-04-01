"""
Grasp policy manager: unified interface for grasp candidate generation.

Supports both heuristic and learned grasp policies with transparent
fallback. The manipulation stack always calls GraspPolicyManager,
never the individual policies directly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from board_state.board_model import BoardModel
from chess_core.interfaces import (
    PIECE_GEOMETRY,
    GraspCandidate,
    PieceType,
    Square,
)

logger = logging.getLogger(__name__)


@dataclass
class GraspPolicyConfig:
    """Configuration for the grasp policy manager."""
    use_learned_grasp: bool = False
    grasp_model_path: str = ""
    default_grasp_ratio: float = 0.65     # fraction of piece height to grasp at
    approach_clearance_m: float = 0.05    # clearance above piece top
    max_candidates: int = 5              # max grasp candidates to generate
    min_neighbor_clearance_m: float = 0.02  # min clearance from neighboring pieces


class GraspPolicyManager:
    """
    Manages grasp candidate generation with learned/heuristic fallback.

    This is the SINGLE POINT where learned policies interact with the
    classical manipulation stack. Disabling learning here disconnects
    the entire RL pipeline without affecting anything else.

    Usage:
        policy = GraspPolicyManager(board_model, config)
        candidates = policy.get_grasp_candidates(
            square=Square.from_algebraic("e2"),
            piece_type=PieceType.PAWN,
            occupied_squares={"d2", "f2", "e3"},
        )
        best_grasp = candidates[0]  # already sorted by score
    """

    def __init__(
        self,
        board: BoardModel,
        config: GraspPolicyConfig | None = None,
    ) -> None:
        self.board = board
        self.config = config or GraspPolicyConfig()
        self._learned_policy = None

        if self.config.use_learned_grasp:
            self._try_load_learned_policy()

    def _try_load_learned_policy(self) -> None:
        """Attempt to load a learned grasp scoring model."""
        try:
            from learning.interfaces import LearnedPolicy
            self._learned_policy = LearnedPolicy(
                model_path=self.config.grasp_model_path,
                fallback_name="heuristic_grasp",
            )
            if self._learned_policy.is_available():
                logger.info("Learned grasp policy loaded successfully")
            else:
                logger.warning("Learned grasp policy unavailable, using heuristic")
                self._learned_policy = None
        except Exception as e:
            logger.warning(f"Failed to load learned grasp policy: {e}")
            self._learned_policy = None

    def get_grasp_candidates(
        self,
        square: Square,
        piece_type: PieceType,
        occupied_squares: set[str] | None = None,
    ) -> list[GraspCandidate]:
        """
        Generate ranked grasp candidates for a piece.

        Args:
            square: Square where the piece is located.
            piece_type: Type of piece to grasp.
            occupied_squares: Set of algebraic names of occupied neighbor squares.

        Returns:
            List of GraspCandidate objects, sorted by score (highest first).
        """
        # Generate heuristic candidates
        candidates = self._generate_heuristic_candidates(
            square, piece_type, occupied_squares
        )

        # Optionally score with learned model
        if self._learned_policy is not None and self._learned_policy.is_available():
            candidates = self._score_with_learned_model(candidates, square, occupied_squares)

        # Sort by score (highest first)
        candidates.sort(key=lambda c: c.score, reverse=True)

        # Limit count
        candidates = candidates[:self.config.max_candidates]

        logger.debug(
            f"Generated {len(candidates)} grasp candidates for "
            f"{piece_type.name} at {square.algebraic}, "
            f"best score={candidates[0].score:.3f}" if candidates else "none"
        )
        return candidates

    def _generate_heuristic_candidates(
        self,
        square: Square,
        piece_type: PieceType,
        occupied_squares: set[str] | None = None,
    ) -> list[GraspCandidate]:
        """Generate grasp candidates using heuristic rules."""
        center = self.board.get_square_center(square)
        geom = PIECE_GEOMETRY[piece_type]
        piece_height_m = geom["height_mm"] / 1000.0
        piece_radius_m = geom["radius_mm"] / 1000.0
        grip_width_mm = geom["grip_width_mm"]

        grasp_z = piece_height_m * self.config.default_grasp_ratio
        approach_height_mm = piece_height_m * 1000.0

        candidates = []

        # Candidate 1: Direct top-down center grasp (highest priority)
        pose = self._make_top_down_pose(center, grasp_z)
        candidates.append(GraspCandidate(
            pose=pose,
            piece_type=piece_type,
            finger_width_mm=grip_width_mm,
            approach_height_mm=approach_height_mm,
            score=1.0,
            source="heuristic_center",
        ))

        # Candidate 2-5: Slightly offset grasps for robustness
        offsets = [
            (0.002, 0.0),   # +X
            (-0.002, 0.0),  # -X
            (0.0, 0.002),   # +Y
            (0.0, -0.002),  # -Y
        ]

        for i, (dx, dy) in enumerate(offsets):
            offset_pos = center.copy()
            offset_pos[0] += dx
            offset_pos[1] += dy

            # Check clearance from neighbors
            score = 0.8
            if occupied_squares:
                score = self._score_neighbor_clearance(
                    offset_pos, square, occupied_squares
                )

            pose = self._make_top_down_pose(offset_pos, grasp_z)
            candidates.append(GraspCandidate(
                pose=pose,
                piece_type=piece_type,
                finger_width_mm=grip_width_mm,
                approach_height_mm=approach_height_mm,
                score=score,
                source=f"heuristic_offset_{i}",
            ))

        return candidates

    def _score_neighbor_clearance(
        self,
        grasp_pos: np.ndarray,
        source_square: Square,
        occupied_squares: set[str],
    ) -> float:
        """
        Score a grasp position based on clearance from neighboring pieces.
        Higher score = more clearance = safer.
        """
        min_clearance = float('inf')

        for sq_name in occupied_squares:
            try:
                sq = Square.from_algebraic(sq_name)
                if sq == source_square:
                    continue
                neighbor_pos = self.board.get_square_center(sq)
                dist = np.linalg.norm(grasp_pos[:2] - neighbor_pos[:2])
                min_clearance = min(min_clearance, dist)
            except ValueError:
                continue

        if min_clearance == float('inf'):
            return 0.9  # no neighbors, good

        # Score: 0 at min_clearance=0, 1.0 at 3 squares away
        sq_size = self.board.config.square_size_m
        normalized = min(min_clearance / (3 * sq_size), 1.0)
        return 0.5 + 0.5 * normalized

    def _score_with_learned_model(
        self,
        candidates: list[GraspCandidate],
        square: Square,
        occupied_squares: set[str] | None,
    ) -> list[GraspCandidate]:
        """Re-score candidates using the learned model."""
        # This would feed each candidate through the learned scoring network
        # For now, just multiply heuristic score with a learned modifier
        logger.debug("Scoring grasps with learned model")

        scored = []
        for c in candidates:
            obs = {
                "grasp_pose": c.pose,
                "piece_type": c.piece_type.value,
                "finger_width": c.finger_width_mm,
                "square_file": square.file,
                "square_rank": square.rank,
            }

            try:
                pred = self._learned_policy.predict(obs)
                learned_score = pred.get("score", c.score)
                c_new = GraspCandidate(
                    pose=c.pose,
                    piece_type=c.piece_type,
                    finger_width_mm=c.finger_width_mm,
                    approach_height_mm=c.approach_height_mm,
                    score=learned_score,
                    source=f"learned:{c.source}",
                )
                scored.append(c_new)
            except Exception:
                scored.append(c)

        return scored

    @staticmethod
    def _make_top_down_pose(position: np.ndarray, height: float) -> np.ndarray:
        """Create a top-down grasp pose at the given position and height."""
        pose = np.eye(4, dtype=np.float64)
        # Rotation: pointing downward
        pose[0, 0] = 1.0
        pose[1, 1] = -1.0
        pose[2, 2] = -1.0
        pose[0, 3] = position[0]
        pose[1, 3] = position[1]
        pose[2, 3] = height
        return pose
