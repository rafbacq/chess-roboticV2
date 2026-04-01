"""
Game manager: top-level orchestrator for chess game state.

Maintains the canonical board position, communicates with Stockfish,
tracks game history, and coordinates between the chess logic layer
and the physical execution layer.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import chess

from chess_core.engine import EngineConfig, EngineEvaluation, StockfishEngine
from chess_core.interfaces import ChessMove, PieceColor, Square
from chess_core.move_parser import parse_uci_move

logger = logging.getLogger(__name__)


class GamePhase(Enum):
    """Current phase of the game state machine."""
    IDLE = auto()              # No game in progress
    AWAITING_MOVE = auto()     # Waiting for a move (human or engine)
    THINKING = auto()          # Engine is computing
    MOVE_VALIDATED = auto()    # Move validated, ready for execution
    EXECUTING = auto()         # Robot is executing the move
    VERIFYING = auto()         # Post-move verification in progress
    GAME_OVER = auto()         # Game has ended
    ERROR = auto()             # Error state requiring intervention


class PlayerType(Enum):
    """Type of player."""
    HUMAN = auto()
    ENGINE = auto()
    REMOTE = auto()


@dataclass
class GameConfig:
    """Configuration for a chess game."""
    white_player: PlayerType = PlayerType.HUMAN
    black_player: PlayerType = PlayerType.ENGINE
    engine_config: EngineConfig = field(default_factory=EngineConfig)
    starting_fen: str = chess.STARTING_FEN
    time_control_s: Optional[float] = None


@dataclass
class MoveRecord:
    """Record of a played move with timestamps."""
    move: ChessMove
    fen_before: str
    fen_after: str
    move_number: int
    timestamp: float
    think_time_s: float = 0.0
    execution_time_s: float = 0.0
    engine_eval: Optional[EngineEvaluation] = None


class GameManager:
    """
    Manages the chess game lifecycle.

    Responsibilities:
        - Maintain the canonical board state via python-chess
        - Interface with Stockfish for engine moves
        - Validate all moves before physical execution
        - Track full game history with timestamps
        - Detect game-over conditions
        - Synchronize internal state with physical board state

    Usage:
        gm = GameManager(GameConfig())
        gm.start_game()

        # For engine move:
        chess_move = gm.get_engine_move()

        # For human move:
        chess_move = gm.validate_and_parse_move("e2e4")

        # After physical execution succeeds:
        gm.confirm_move(chess_move)
    """

    def __init__(self, config: GameConfig | None = None) -> None:
        self.config = config or GameConfig()
        self._board = chess.Board(self.config.starting_fen)
        self._engine = StockfishEngine(self.config.engine_config)
        self._history: list[MoveRecord] = []
        self._phase = GamePhase.IDLE
        self._game_start_time: float = 0.0

    @property
    def phase(self) -> GamePhase:
        return self._phase

    @property
    def board(self) -> chess.Board:
        """The current python-chess board (read-only reference)."""
        return self._board

    @property
    def fen(self) -> str:
        return self._board.fen()

    @property
    def current_color(self) -> PieceColor:
        return PieceColor.WHITE if self._board.turn == chess.WHITE else PieceColor.BLACK

    @property
    def current_player_type(self) -> PlayerType:
        if self._board.turn == chess.WHITE:
            return self.config.white_player
        return self.config.black_player

    @property
    def move_number(self) -> int:
        return self._board.fullmove_number

    @property
    def history(self) -> list[MoveRecord]:
        return list(self._history)

    def start_game(self) -> None:
        """Initialize a new game."""
        self._board = chess.Board(self.config.starting_fen)
        self._history.clear()
        self._game_start_time = time.time()

        self._engine.start()
        self._engine.new_game()

        self._phase = GamePhase.AWAITING_MOVE
        logger.info(
            f"Game started. White: {self.config.white_player.name}, "
            f"Black: {self.config.black_player.name}"
        )

    def stop_game(self) -> None:
        """Stop the current game and clean up."""
        self._engine.stop()
        self._phase = GamePhase.IDLE
        logger.info(f"Game stopped after {len(self._history)} moves")

    def validate_and_parse_move(self, uci_string: str) -> ChessMove:
        """
        Validate a UCI move string against current position and return
        a fully classified ChessMove.

        Args:
            uci_string: UCI move string like "e2e4".

        Returns:
            Validated ChessMove ready for physical execution.

        Raises:
            ValueError: If the move is illegal.
            RuntimeError: If the game is not in AWAITING_MOVE phase.
        """
        if self._phase != GamePhase.AWAITING_MOVE:
            raise RuntimeError(
                f"Cannot validate move in phase {self._phase.name}. "
                f"Expected AWAITING_MOVE."
            )

        chess_move = parse_uci_move(uci_string, self._board)
        self._phase = GamePhase.MOVE_VALIDATED
        logger.info(f"Move validated: {chess_move}")
        return chess_move

    def get_engine_move(self) -> ChessMove:
        """
        Ask Stockfish for the best move in the current position.

        Returns:
            ChessMove chosen by the engine.

        Raises:
            RuntimeError: If the game is not in AWAITING_MOVE phase.
        """
        if self._phase != GamePhase.AWAITING_MOVE:
            raise RuntimeError(
                f"Cannot get engine move in phase {self._phase.name}. "
                f"Expected AWAITING_MOVE."
            )

        self._phase = GamePhase.THINKING
        logger.info("Engine thinking...")

        # Set position in engine
        moves_uci = [record.move.uci_string for record in self._history]
        self._engine.set_position_from_fen(self.config.starting_fen, moves_uci)

        # Get best move
        t0 = time.time()
        evaluation = self._engine.get_best_move()
        think_time = time.time() - t0

        # Parse the engine's move
        chess_move = parse_uci_move(evaluation.best_move, self._board)

        self._phase = GamePhase.MOVE_VALIDATED
        logger.info(
            f"Engine chose: {chess_move} "
            f"(score: {evaluation.score_cp}cp, think: {think_time:.1f}s)"
        )
        return chess_move

    def confirm_move(self, move: ChessMove, execution_time_s: float = 0.0) -> None:
        """
        Confirm that a move has been physically executed and verified.
        Updates the internal board state.

        Args:
            move: The ChessMove that was executed.
            execution_time_s: Time taken for physical execution.

        Raises:
            RuntimeError: If the game is not in MOVE_VALIDATED or VERIFYING phase.
        """
        if self._phase not in (GamePhase.MOVE_VALIDATED, GamePhase.VERIFYING):
            raise RuntimeError(
                f"Cannot confirm move in phase {self._phase.name}. "
                f"Expected MOVE_VALIDATED or VERIFYING."
            )

        fen_before = self._board.fen()

        # Apply move to internal board
        chess_move = chess.Move.from_uci(move.uci_string)
        self._board.push(chess_move)

        # Record
        record = MoveRecord(
            move=move,
            fen_before=fen_before,
            fen_after=self._board.fen(),
            move_number=self._board.fullmove_number,
            timestamp=time.time(),
            execution_time_s=execution_time_s,
        )
        self._history.append(record)

        # Check game over
        if self._board.is_game_over():
            self._phase = GamePhase.GAME_OVER
            result = self._board.result()
            logger.info(f"Game over: {result}")
        else:
            self._phase = GamePhase.AWAITING_MOVE
            logger.info(
                f"Move {len(self._history)} confirmed: {move}. "
                f"Now {self.current_color.name}'s turn."
            )

    def set_phase(self, phase: GamePhase) -> None:
        """Manually transition the game phase (for execution pipeline use)."""
        old = self._phase
        self._phase = phase
        logger.debug(f"Phase transition: {old.name} → {phase.name}")

    def get_piece_at(self, square: Square) -> Optional[tuple]:
        """Get the piece at a square, returns (PieceType, PieceColor) or None."""
        from chess_core.interfaces import PieceType, PieceColor

        sq_idx = chess.square(square.file, square.rank)
        piece = self._board.piece_at(sq_idx)
        if piece is None:
            return None

        piece_map = {
            chess.PAWN: PieceType.PAWN,
            chess.KNIGHT: PieceType.KNIGHT,
            chess.BISHOP: PieceType.BISHOP,
            chess.ROOK: PieceType.ROOK,
            chess.QUEEN: PieceType.QUEEN,
            chess.KING: PieceType.KING,
        }
        color_map = {
            chess.WHITE: PieceColor.WHITE,
            chess.BLACK: PieceColor.BLACK,
        }
        return piece_map[piece.piece_type], color_map[piece.color]

    def get_expected_occupancy(self) -> dict[str, bool]:
        """
        Get expected occupancy of all 64 squares based on internal board state.

        Returns:
            Dict mapping algebraic square names to occupancy (True = piece present).
        """
        occupancy = {}
        for sq in chess.SQUARES:
            file = chess.square_file(sq)
            rank = chess.square_rank(sq)
            alg = f"{chr(ord('a') + file)}{rank + 1}"
            occupancy[alg] = self._board.piece_at(sq) is not None
        return occupancy

    def check_board_state_consistency(
        self, observed_occupancy: dict[str, bool]
    ) -> list[str]:
        """
        Compare expected vs observed board occupancy.

        Args:
            observed_occupancy: Dict mapping algebraic squares to observed occupancy.

        Returns:
            List of discrepancy descriptions. Empty list = consistent.
        """
        expected = self.get_expected_occupancy()
        discrepancies = []

        for sq_name in expected:
            if sq_name not in observed_occupancy:
                continue
            exp = expected[sq_name]
            obs = observed_occupancy[sq_name]
            if exp and not obs:
                discrepancies.append(f"{sq_name}: expected occupied, observed empty")
            elif not exp and obs:
                discrepancies.append(f"{sq_name}: expected empty, observed occupied")

        if discrepancies:
            logger.warning(
                f"Board state inconsistency: {len(discrepancies)} discrepancies"
            )
            for d in discrepancies:
                logger.warning(f"  - {d}")

        return discrepancies

    def __repr__(self) -> str:
        return (
            f"GameManager(phase={self._phase.name}, "
            f"move={self.move_number}, "
            f"turn={self.current_color.name})"
        )
