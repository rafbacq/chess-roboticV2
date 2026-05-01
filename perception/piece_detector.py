"""
Piece detector: detect piece occupancy and optionally classify pieces.

Works on a warped top-down board image (from BoardDetector).
Uses classical CV (color thresholding, contour analysis) for the
occupancy baseline, with hooks for learned classification later.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from chess_core.interfaces import PieceColor, PieceType

logger = logging.getLogger(__name__)


@dataclass
class SquareAnalysis:
    """Analysis result for a single square."""
    square_name: str
    is_occupied: bool
    confidence: float = 0.0
    piece_color: Optional[PieceColor] = None
    piece_type: Optional[PieceType] = None
    center_offset_px: tuple[float, float] = (0.0, 0.0)  # offset from square center
    roi: Optional[np.ndarray] = None  # cropped square image


@dataclass
class BoardAnalysis:
    """Full board occupancy analysis."""
    squares: dict[str, SquareAnalysis] = field(default_factory=dict)
    timestamp: float = 0.0
    confidence: float = 0.0
    image_shape: tuple[int, ...] = (0, 0)

    @property
    def occupied_squares(self) -> list[str]:
        return [name for name, sq in self.squares.items() if sq.is_occupied]

    @property
    def empty_squares(self) -> list[str]:
        return [name for name, sq in self.squares.items() if not sq.is_occupied]

    def get_occupancy_map(self) -> dict[str, bool]:
        return {name: sq.is_occupied for name, sq in self.squares.items()}


class PieceDetector:
    """
    Detects piece occupancy on the chess board.

    Primary method: compare each square's intensity/color statistics
    against empty-square baselines. An occupied square will have
    significantly different statistics due to the piece.

    Usage:
        detector = PieceDetector()
        detector.calibrate_empty_board(warped_empty_board_image)
        analysis = detector.detect(warped_board_image)
    """

    def __init__(
        self,
        occupancy_threshold: float = 30.0,
        color_threshold: float = 50.0,
        square_margin_frac: float = 0.15,
        classifier_model_path: str = "",
    ) -> None:
        self._occupancy_threshold = occupancy_threshold
        self._color_threshold = color_threshold
        self._margin_frac = square_margin_frac
        self._empty_baselines: dict[str, np.ndarray] = {}
        self._classifier = None

        # Load ML classifier if model path is provided
        if classifier_model_path:
            try:
                from perception.piece_classifier import PieceClassifier, ClassifierConfig
                config = ClassifierConfig(model_path=classifier_model_path)
                self._classifier = PieceClassifier(config)
                if self._classifier.load_model():
                    logger.info(f"Loaded piece classifier: {classifier_model_path}")
                else:
                    self._classifier = None
                    logger.warning("Failed to load piece classifier, using classical only")
            except ImportError:
                logger.warning("torch not available, using classical detection only")

    def calibrate_empty_board(self, warped_image: np.ndarray) -> None:
        """
        Capture baselines from an empty board image.

        Args:
            warped_image: Top-down warped board image (from BoardDetector).
        """
        for rank in range(8):
            for file in range(8):
                sq_name = f"{chr(ord('a') + file)}{rank + 1}"
                roi = self._extract_square_roi(warped_image, file, rank)
                self._empty_baselines[sq_name] = np.mean(roi, axis=(0, 1))

        logger.info(f"Calibrated empty board baselines for {len(self._empty_baselines)} squares")

    def detect(self, warped_image: np.ndarray) -> BoardAnalysis:
        """
        Detect piece occupancy on all 64 squares.

        Args:
            warped_image: Top-down warped board image.

        Returns:
            BoardAnalysis with per-square results.
        """
        analysis = BoardAnalysis(image_shape=warped_image.shape)
        confidences = []

        for rank in range(8):
            for file in range(8):
                sq_name = f"{chr(ord('a') + file)}{rank + 1}"
                roi = self._extract_square_roi(warped_image, file, rank)

                sq_analysis = self._analyze_square(sq_name, roi)
                analysis.squares[sq_name] = sq_analysis
                confidences.append(sq_analysis.confidence)

        analysis.confidence = float(np.mean(confidences)) if confidences else 0.0
        return analysis

    def _analyze_square(self, sq_name: str, roi: np.ndarray) -> SquareAnalysis:
        """Analyze a single square for piece occupancy."""
        mean_color = np.mean(roi, axis=(0, 1))

        # Method 1: Compare against empty baseline
        if sq_name in self._empty_baselines:
            baseline = self._empty_baselines[sq_name]
            diff = np.linalg.norm(mean_color - baseline)
            is_occupied = diff > self._occupancy_threshold
            confidence = min(diff / (self._occupancy_threshold * 3), 1.0)
        else:
            # Without baseline, use center region analysis
            h, w = roi.shape[:2]
            center_region = roi[h // 4:3 * h // 4, w // 4:3 * w // 4]
            edge_region_mean = np.mean(roi, axis=(0, 1))
            center_mean = np.mean(center_region, axis=(0, 1))
            diff = np.linalg.norm(center_mean - edge_region_mean)
            is_occupied = diff > self._color_threshold
            confidence = min(diff / (self._color_threshold * 3), 1.0)

        # Try to determine piece color using classical heuristics
        piece_color = None
        piece_type = None
        if is_occupied:
            gray_val = np.mean(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))
            # Rough heuristic: lighter pieces = white, darker = black
            piece_color = PieceColor.WHITE if gray_val > 128 else PieceColor.BLACK

        # Enhance with ML classifier when available
        if self._classifier is not None:
            from perception.piece_classifier import PieceClass
            pred_class, ml_conf = self._classifier.classify(roi)
            if ml_conf > 0.5:  # Only trust high-confidence predictions
                if pred_class == PieceClass.EMPTY:
                    is_occupied = False
                    confidence = ml_conf
                    piece_color = None
                    piece_type = None
                else:
                    is_occupied = True
                    confidence = ml_conf
                    # Map PieceClass to PieceColor + PieceType
                    _type_map = {
                        1: PieceType.PAWN, 2: PieceType.KNIGHT, 3: PieceType.BISHOP,
                        4: PieceType.ROOK, 5: PieceType.QUEEN, 6: PieceType.KING,
                        7: PieceType.PAWN, 8: PieceType.KNIGHT, 9: PieceType.BISHOP,
                        10: PieceType.ROOK, 11: PieceType.QUEEN, 12: PieceType.KING,
                    }
                    piece_color = PieceColor.WHITE if pred_class.value <= 6 else PieceColor.BLACK
                    piece_type = _type_map.get(pred_class.value)

        return SquareAnalysis(
            square_name=sq_name,
            is_occupied=is_occupied,
            confidence=confidence,
            piece_color=piece_color,
            piece_type=piece_type,
            roi=roi,
        )

    def _extract_square_roi(
        self,
        warped_image: np.ndarray,
        file: int,
        rank: int,
    ) -> np.ndarray:
        """
        Extract the region of interest for a single square from the warped image.

        The warped image has a1 at bottom-left, h8 at top-right.
        Image Y increases downward, so rank 7 (=8) is at the top.
        """
        h, w = warped_image.shape[:2]
        sq_w = w / 8
        sq_h = h / 8

        # Image coordinates: rank 7 at top (y=0), rank 0 at bottom
        x1 = int(file * sq_w)
        x2 = int((file + 1) * sq_w)
        y1 = int((7 - rank) * sq_h)  # flip for image coords
        y2 = int((8 - rank) * sq_h)

        # Apply margin to avoid square edges
        margin_x = int(sq_w * self._margin_frac)
        margin_y = int(sq_h * self._margin_frac)
        x1 += margin_x
        x2 -= margin_x
        y1 += margin_y
        y2 -= margin_y

        return warped_image[y1:y2, x1:x2]

    def draw_occupancy(
        self,
        warped_image: np.ndarray,
        analysis: BoardAnalysis,
    ) -> np.ndarray:
        """Draw occupancy overlay on warped board image."""
        vis = warped_image.copy()
        h, w = vis.shape[:2]
        sq_w = w // 8
        sq_h = h // 8

        for rank in range(8):
            for file in range(8):
                sq_name = f"{chr(ord('a') + file)}{rank + 1}"
                sq = analysis.squares.get(sq_name)
                if sq is None:
                    continue

                x = file * sq_w + sq_w // 2
                y = (7 - rank) * sq_h + sq_h // 2

                if sq.is_occupied:
                    color = (0, 255, 0)  # green
                    cv2.circle(vis, (x, y), sq_w // 4, color, 2)
                else:
                    color = (100, 100, 100)

                # Confidence text
                cv2.putText(vis, f"{sq.confidence:.1f}", (x - 10, y + sq_h // 3),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

        return vis
