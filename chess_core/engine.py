"""
Stockfish UCI engine wrapper.

Provides a clean Python interface to communicate with Stockfish via the
Universal Chess Interface (UCI) protocol. Handles process lifecycle,
move requests, evaluation, and engine configuration.
"""

from __future__ import annotations

import logging
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class EngineConfig:
    """Configuration for the Stockfish engine."""
    stockfish_path: str = "stockfish"
    depth: int = 15
    time_limit_ms: int = 2000
    threads: int = 2
    hash_mb: int = 256
    skill_level: int = 20  # 0-20, 20 is strongest
    elo_limit: Optional[int] = None  # if set, limits playing strength


@dataclass
class EngineEvaluation:
    """Result of a position evaluation."""
    best_move: str  # UCI format, e.g. "e2e4"
    ponder_move: Optional[str] = None
    score_cp: Optional[int] = None  # centipawns, from side-to-move perspective
    score_mate: Optional[int] = None  # mate in N, positive = winning
    depth: int = 0
    nodes: int = 0
    time_ms: int = 0
    pv: list[str] = field(default_factory=list)  # principal variation


class StockfishEngine:
    """
    Interface to the Stockfish chess engine via UCI protocol.

    Usage:
        engine = StockfishEngine(EngineConfig(stockfish_path="/usr/bin/stockfish"))
        engine.start()
        engine.set_position_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        result = engine.get_best_move()
        print(result.best_move)  # e.g. "e2e4"
        engine.stop()
    """

    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()
        self._process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._is_ready = False

    def start(self) -> None:
        """Start the Stockfish process and initialize UCI."""
        path = Path(self.config.stockfish_path)
        logger.info(f"Starting Stockfish engine at: {path}")

        try:
            self._process = subprocess.Popen(
                [str(path)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except FileNotFoundError:
            raise RuntimeError(
                f"Stockfish not found at '{path}'. "
                "Install Stockfish and set the correct path in config."
            )

        # Initialize UCI
        self._send("uci")
        self._wait_for("uciok")

        # Set options
        self._send(f"setoption name Threads value {self.config.threads}")
        self._send(f"setoption name Hash value {self.config.hash_mb}")

        if self.config.elo_limit is not None:
            self._send("setoption name UCI_LimitStrength value true")
            self._send(f"setoption name UCI_Elo value {self.config.elo_limit}")
        else:
            self._send(f"setoption name Skill Level value {self.config.skill_level}")

        # Confirm ready
        self._send("isready")
        self._wait_for("readyok")
        self._is_ready = True
        logger.info("Stockfish engine initialized successfully")

    def stop(self) -> None:
        """Stop the Stockfish process."""
        if self._process:
            self._send("quit")
            self._process.wait(timeout=5)
            self._process = None
            self._is_ready = False
            logger.info("Stockfish engine stopped")

    @property
    def is_ready(self) -> bool:
        return self._is_ready and self._process is not None

    def set_position_startpos(self, moves: list[str] | None = None) -> None:
        """Set position to starting position, optionally with moves played."""
        cmd = "position startpos"
        if moves:
            cmd += " moves " + " ".join(moves)
        self._send(cmd)

    def set_position_from_fen(self, fen: str, moves: list[str] | None = None) -> None:
        """Set position from FEN string, optionally with additional moves."""
        cmd = f"position fen {fen}"
        if moves:
            cmd += " moves " + " ".join(moves)
        self._send(cmd)

    def get_best_move(
        self,
        depth: int | None = None,
        time_limit_ms: int | None = None,
    ) -> EngineEvaluation:
        """
        Get the best move for the current position.

        Args:
            depth: Search depth override (uses config default if None).
            time_limit_ms: Time limit override in milliseconds.

        Returns:
            EngineEvaluation with best move and analysis details.
        """
        with self._lock:
            d = depth or self.config.depth
            t = time_limit_ms or self.config.time_limit_ms

            self._send(f"go depth {d} movetime {t}")

            evaluation = EngineEvaluation(best_move="")
            while True:
                line = self._read_line()
                if line is None:
                    break

                if line.startswith("info depth"):
                    evaluation = self._parse_info_line(line, evaluation)
                elif line.startswith("bestmove"):
                    parts = line.split()
                    evaluation.best_move = parts[1]
                    if len(parts) >= 4 and parts[2] == "ponder":
                        evaluation.ponder_move = parts[3]
                    break

            if not evaluation.best_move:
                raise RuntimeError("Stockfish did not return a best move")

            logger.debug(
                f"Best move: {evaluation.best_move} "
                f"(score: {evaluation.score_cp}cp, depth: {evaluation.depth})"
            )
            return evaluation

    def is_move_legal(self, fen: str, move_uci: str) -> bool:
        """
        Check if a move is legal in the given position.

        Uses python-chess for legality checking rather than Stockfish,
        since Stockfish doesn't have a direct legality check command.
        """
        import chess
        board = chess.Board(fen)
        try:
            m = chess.Move.from_uci(move_uci)
            return m in board.legal_moves
        except (ValueError, chess.InvalidMoveError):
            return False

    def new_game(self) -> None:
        """Signal a new game to clear Stockfish's internal state."""
        self._send("ucinewgame")
        self._send("isready")
        self._wait_for("readyok")

    def _send(self, command: str) -> None:
        """Send a command to the Stockfish process."""
        if not self._process or not self._process.stdin:
            raise RuntimeError("Engine process not running")
        logger.debug(f"UCI >>> {command}")
        self._process.stdin.write(command + "\n")
        self._process.stdin.flush()

    def _read_line(self) -> Optional[str]:
        """Read a line from Stockfish stdout."""
        if not self._process or not self._process.stdout:
            return None
        line = self._process.stdout.readline().strip()
        if line:
            logger.debug(f"UCI <<< {line}")
        return line

    def _wait_for(self, target: str, timeout_lines: int = 1000) -> None:
        """Read lines until we see one starting with target."""
        for _ in range(timeout_lines):
            line = self._read_line()
            if line and line.startswith(target):
                return
        raise RuntimeError(f"Timeout waiting for '{target}' from Stockfish")

    @staticmethod
    def _parse_info_line(line: str, eval_so_far: EngineEvaluation) -> EngineEvaluation:
        """Parse a 'info' line from Stockfish into evaluation data."""
        tokens = line.split()
        result = EngineEvaluation(best_move=eval_so_far.best_move)

        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token == "depth" and i + 1 < len(tokens):
                result.depth = int(tokens[i + 1])
                i += 2
            elif token == "nodes" and i + 1 < len(tokens):
                result.nodes = int(tokens[i + 1])
                i += 2
            elif token == "time" and i + 1 < len(tokens):
                result.time_ms = int(tokens[i + 1])
                i += 2
            elif token == "score" and i + 2 < len(tokens):
                if tokens[i + 1] == "cp":
                    result.score_cp = int(tokens[i + 2])
                    result.score_mate = None
                elif tokens[i + 1] == "mate":
                    result.score_mate = int(tokens[i + 2])
                    result.score_cp = None
                i += 3
            elif token == "pv":
                result.pv = tokens[i + 1:]
                break
            else:
                i += 1

        return result

    def __enter__(self) -> StockfishEngine:
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()

    def __del__(self) -> None:
        try:
            self.stop()
        except Exception:
            pass
