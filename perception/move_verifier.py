"""
Move verifier: visual verification of chess moves after physical execution.

Compares before/after images to confirm:
    1. Source square is now empty
    2. Target square is now occupied
    3. No unexpected changes elsewhere

Provides confidence scores and diagnostic images for debugging.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from chess_core.interfaces import ChessMove, MoveType, VerificationResult

logger = logging.getLogger(__name__)


@dataclass
class VerificationConfig:
    """Configuration for move verification."""
    min_confidence: float = 0.7         # minimum confidence to accept
    change_threshold: float = 25.0      # pixel intensity change threshold
    diagnostic_image_dir: str = "logs/verification"  # save diagnostic images
    max_unexpected_changes: int = 2     # max unexpected square changes allowed
    save_diagnostics: bool = True


class MoveVerifier:
    """
    Verifies chess move execution by comparing before/after camera images.

    Uses difference images and per-square analysis to determine whether
    the physical execution matches the expected move.

    Usage:
        verifier = MoveVerifier(config)
        # Before the move:
        verifier.capture_before(warped_board_image)
        # After the move:
        result = verifier.verify(warped_board_image_after, expected_move)
    """

    def __init__(self, config: VerificationConfig | None = None) -> None:
        self.config = config or VerificationConfig()
        self._before_image: Optional[np.ndarray] = None
        self._before_occupancy: Optional[dict[str, bool]] = None

    def capture_before(
        self,
        warped_image: np.ndarray,
        occupancy: Optional[dict[str, bool]] = None,
    ) -> None:
        """
        Capture the board state BEFORE move execution.

        Args:
            warped_image: Warped top-down board image.
            occupancy: Optional known occupancy map.
        """
        self._before_image = warped_image.copy()
        self._before_occupancy = occupancy

    def verify(
        self,
        warped_image_after: np.ndarray,
        move: ChessMove,
    ) -> VerificationResult:
        """
        Verify that a move was executed correctly.

        Args:
            warped_image_after: Warped board image AFTER move execution.
            move: The chess move that was executed.

        Returns:
            VerificationResult with success/failure and diagnostics.
        """
        if self._before_image is None:
            logger.warning("No before-image captured, cannot verify")
            return VerificationResult(
                success=False,
                mismatch_details="No before-image available",
            )

        before_gray = cv2.cvtColor(self._before_image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        after_gray = cv2.cvtColor(warped_image_after, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Compute per-square change magnitude
        source_change = self._get_square_change(before_gray, after_gray, move.source.algebraic)
        target_change = self._get_square_change(before_gray, after_gray, move.target.algebraic)

        # Check expected changes
        source_valid = source_change > self.config.change_threshold  # should have changed (piece left)
        target_valid = target_change > self.config.change_threshold  # should have changed (piece arrived)

        # For captures, en passant, castling: check additional squares
        additional_issues = []
        if move.move_type == MoveType.EN_PASSANT:
            from chess_core.move_parser import get_en_passant_capture_square
            cap_sq = get_en_passant_capture_square(move)
            cap_change = self._get_square_change(before_gray, after_gray, cap_sq.algebraic)
            if cap_change < self.config.change_threshold:
                additional_issues.append(f"EP capture square {cap_sq.algebraic} unchanged")

        if move.is_castling:
            from chess_core.move_parser import get_castling_rook_move
            rook_src, rook_tgt = get_castling_rook_move(move)
            rook_src_change = self._get_square_change(before_gray, after_gray, rook_src.algebraic)
            rook_tgt_change = self._get_square_change(before_gray, after_gray, rook_tgt.algebraic)
            if rook_src_change < self.config.change_threshold:
                additional_issues.append(f"Rook source {rook_src.algebraic} unchanged")
            if rook_tgt_change < self.config.change_threshold:
                additional_issues.append(f"Rook target {rook_tgt.algebraic} unchanged")

        # Compute overall confidence
        confidence = (min(source_change, 100) + min(target_change, 100)) / 200.0

        # Check for unexpected changes
        unexpected = self._count_unexpected_changes(
            before_gray, after_gray, move
        )

        success = (
            source_valid
            and target_valid
            and len(additional_issues) == 0
            and unexpected <= self.config.max_unexpected_changes
            and confidence >= self.config.min_confidence
        )

        # Save diagnostic image
        diag_path = ""
        if self.config.save_diagnostics:
            diag_path = self._save_diagnostic(
                self._before_image, warped_image_after, move, success
            )

        mismatch = ""
        if not success:
            issues = []
            if not source_valid:
                issues.append(f"Source {move.source.algebraic} appears unchanged")
            if not target_valid:
                issues.append(f"Target {move.target.algebraic} appears unchanged")
            issues.extend(additional_issues)
            if unexpected > self.config.max_unexpected_changes:
                issues.append(f"{unexpected} unexpected square changes detected")
            mismatch = "; ".join(issues)

        result = VerificationResult(
            success=success,
            source_empty=source_valid,
            target_occupied=target_valid,
            confidence=confidence,
            diagnostic_image_path=diag_path,
            mismatch_details=mismatch,
        )

        if success:
            logger.info(f"Move verified: {move} (confidence={confidence:.2f})")
        else:
            logger.warning(f"Move verification FAILED: {move} — {mismatch}")

        return result

    def _get_square_change(
        self,
        before_gray: np.ndarray,
        after_gray: np.ndarray,
        square_name: str,
    ) -> float:
        """Compute the mean absolute change in a square's ROI."""
        file = ord(square_name[0]) - ord('a')
        rank = int(square_name[1]) - 1

        roi_before = self._extract_roi(before_gray, file, rank)
        roi_after = self._extract_roi(after_gray, file, rank)

        return float(np.mean(np.abs(roi_after - roi_before)))

    def _extract_roi(
        self,
        gray_image: np.ndarray,
        file: int,
        rank: int,
    ) -> np.ndarray:
        """Extract a square's ROI from the warped image."""
        h, w = gray_image.shape[:2]
        sq_w = w / 8
        sq_h = h / 8
        margin = 0.15

        x1 = int(file * sq_w + sq_w * margin)
        x2 = int((file + 1) * sq_w - sq_w * margin)
        y1 = int((7 - rank) * sq_h + sq_h * margin)
        y2 = int((8 - rank) * sq_h - sq_h * margin)

        return gray_image[y1:y2, x1:x2]

    def _count_unexpected_changes(
        self,
        before_gray: np.ndarray,
        after_gray: np.ndarray,
        move: ChessMove,
    ) -> int:
        """Count squares with unexpected changes (not part of the move)."""
        expected_changes = {move.source.algebraic, move.target.algebraic}

        if move.move_type == MoveType.EN_PASSANT:
            from chess_core.move_parser import get_en_passant_capture_square
            expected_changes.add(get_en_passant_capture_square(move).algebraic)

        if move.is_castling:
            from chess_core.move_parser import get_castling_rook_move
            rs, rt = get_castling_rook_move(move)
            expected_changes.add(rs.algebraic)
            expected_changes.add(rt.algebraic)

        unexpected = 0
        for rank in range(8):
            for file in range(8):
                sq_name = f"{chr(ord('a') + file)}{rank + 1}"
                if sq_name in expected_changes:
                    continue
                change = self._get_square_change(before_gray, after_gray, sq_name)
                if change > self.config.change_threshold:
                    unexpected += 1

        return unexpected

    def _save_diagnostic(
        self,
        before: np.ndarray,
        after: np.ndarray,
        move: ChessMove,
        success: bool,
    ) -> str:
        """Save a side-by-side diagnostic image."""
        diag_dir = Path(self.config.diagnostic_image_dir)
        diag_dir.mkdir(parents=True, exist_ok=True)

        # Create side-by-side
        h = max(before.shape[0], after.shape[0])
        w = before.shape[1] + after.shape[1] + 20
        canvas = np.zeros((h + 40, w, 3), dtype=np.uint8)

        canvas[:before.shape[0], :before.shape[1]] = before
        canvas[:after.shape[0], before.shape[1] + 20:] = after

        status_color = (0, 255, 0) if success else (0, 0, 255)
        status_text = "VERIFIED" if success else "FAILED"
        cv2.putText(canvas, f"{move.uci_string}: {status_text}",
                   (10, h + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)

        filename = f"verify_{move.uci_string}_{int(time.time())}.png"
        filepath = str(diag_dir / filename)
        cv2.imwrite(filepath, canvas)
        return filepath
