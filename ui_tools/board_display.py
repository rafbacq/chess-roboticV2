"""
Board state visualizer and telemetry dashboard.

Provides terminal-based visualization for:
  - Current board state (ASCII)
  - Arm/gripper telemetry (joint positions, EE pose)
  - Move history and verification results
  - Per-move timing breakdown

For use during development, debugging, and live demos.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# =========================================================================
# ASCII Board Renderer
# =========================================================================

PIECE_SYMBOLS = {
    ("KING", "WHITE"): "♔", ("QUEEN", "WHITE"): "♕", ("ROOK", "WHITE"): "♖",
    ("BISHOP", "WHITE"): "♗", ("KNIGHT", "WHITE"): "♘", ("PAWN", "WHITE"): "♙",
    ("KING", "BLACK"): "♚", ("QUEEN", "BLACK"): "♛", ("ROOK", "BLACK"): "♜",
    ("BISHOP", "BLACK"): "♝", ("KNIGHT", "BLACK"): "♞", ("PAWN", "BLACK"): "♟",
}

PIECE_ASCII = {
    ("KING", "WHITE"): "K", ("QUEEN", "WHITE"): "Q", ("ROOK", "WHITE"): "R",
    ("BISHOP", "WHITE"): "B", ("KNIGHT", "WHITE"): "N", ("PAWN", "WHITE"): "P",
    ("KING", "BLACK"): "k", ("QUEEN", "BLACK"): "q", ("ROOK", "BLACK"): "r",
    ("BISHOP", "BLACK"): "b", ("KNIGHT", "BLACK"): "n", ("PAWN", "BLACK"): "p",
}


def render_board_ascii(
    board_fen: str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR",
    use_unicode: bool = False,
    highlight_squares: set[str] | None = None,
) -> str:
    """
    Render a board position as an ASCII string.

    Args:
        board_fen: Board FEN string (piece placement only).
        use_unicode: Use Unicode chess symbols (♔♕♖...).
        highlight_squares: Set of algebraic squares to highlight (e.g., {"e2", "e4"}).

    Returns:
        Multi-line string with the rendered board.
    """
    highlight = highlight_squares or set()
    rows = board_fen.split("/")
    lines = []
    lines.append("  ┌───┬───┬───┬───┬───┬───┬───┬───┐")

    fen_to_piece = {
        'K': ("KING", "WHITE"), 'Q': ("QUEEN", "WHITE"), 'R': ("ROOK", "WHITE"),
        'B': ("BISHOP", "WHITE"), 'N': ("KNIGHT", "WHITE"), 'P': ("PAWN", "WHITE"),
        'k': ("KING", "BLACK"), 'q': ("QUEEN", "BLACK"), 'r': ("ROOK", "BLACK"),
        'b': ("BISHOP", "BLACK"), 'n': ("KNIGHT", "BLACK"), 'p': ("PAWN", "BLACK"),
    }

    for rank_idx, row in enumerate(rows):
        rank_num = 8 - rank_idx
        cells = []
        file_idx = 0
        for ch in row:
            if ch.isdigit():
                for _ in range(int(ch)):
                    sq_name = f"{chr(ord('a') + file_idx)}{rank_num}"
                    marker = "·" if sq_name not in highlight else "*"
                    cells.append(f" {marker} ")
                    file_idx += 1
            else:
                sq_name = f"{chr(ord('a') + file_idx)}{rank_num}"
                piece = fen_to_piece.get(ch, None)
                if piece and use_unicode:
                    sym = PIECE_SYMBOLS.get(piece, "?")
                elif piece:
                    sym = PIECE_ASCII.get(piece, "?")
                else:
                    sym = "?"
                prefix = ">" if sq_name in highlight else " "
                cells.append(f"{prefix}{sym} ")
                file_idx += 1

        line = f"{rank_num} │{'│'.join(cells)}│"
        lines.append(line)
        if rank_idx < 7:
            lines.append("  ├───┼───┼───┼───┼───┼───┼───┼───┤")

    lines.append("  └───┴───┴───┴───┴───┴───┴───┴───┘")
    lines.append("    a   b   c   d   e   f   g   h")
    return "\n".join(lines)


# =========================================================================
# Telemetry Display
# =========================================================================

@dataclass
class TelemetrySummary:
    """Summary of telemetry from one move execution."""
    move_uci: str
    duration_s: float
    n_waypoints: int
    max_joint_velocity_rads: float
    max_ee_velocity_ms: float
    peak_gripper_force_n: float
    stages: list[str]


def format_telemetry_table(records: list[TelemetrySummary]) -> str:
    """Format telemetry records as a table."""
    lines = []
    header = f"{'Move':<8} {'Time':>6} {'Wpts':>5} {'MaxJVel':>8} {'MaxEEVel':>9} {'Force':>6} {'Stages'}"
    lines.append(header)
    lines.append("─" * len(header))

    for r in records:
        stages_str = " → ".join(r.stages[:4])
        if len(r.stages) > 4:
            stages_str += f" (+{len(r.stages) - 4})"
        lines.append(
            f"{r.move_uci:<8} {r.duration_s:>5.2f}s {r.n_waypoints:>5} "
            f"{r.max_joint_velocity_rads:>7.3f} {r.max_ee_velocity_ms:>8.4f} "
            f"{r.peak_gripper_force_n:>5.1f} {stages_str}"
        )

    return "\n".join(lines)


# =========================================================================
# Move History Display
# =========================================================================

def format_move_history(
    moves: list[dict],
    max_display: int = 20,
) -> str:
    """
    Format move history as a numbered table.

    Args:
        moves: List of dicts with keys: uci, status, duration_s, verified.
        max_display: Max moves to show.
    """
    lines = []
    lines.append(f"{'#':>3} {'UCI':<8} {'Status':<12} {'Time':>6} {'Verified'}")
    lines.append("─" * 48)

    display = moves[-max_display:]
    start_idx = len(moves) - len(display) + 1

    for i, m in enumerate(display, start=start_idx):
        status = m.get("status", "?")
        uci = m.get("uci", "?")
        dur = m.get("duration_s", 0)
        verified = "✓" if m.get("verified", False) else "✗"
        lines.append(f"{i:>3} {uci:<8} {status:<12} {dur:>5.2f}s {verified}")

    if len(moves) > max_display:
        lines.append(f"    ... ({len(moves) - max_display} earlier moves hidden)")

    return "\n".join(lines)


# =========================================================================
# Joint State Display
# =========================================================================

def format_joint_state(
    names: list[str],
    positions: np.ndarray,
    velocities: np.ndarray | None = None,
    limits_lower: np.ndarray | None = None,
    limits_upper: np.ndarray | None = None,
) -> str:
    """Format current joint state as a compact table."""
    lines = []
    header = f"{'Joint':<12} {'Position':>10} {'Velocity':>10}"
    if limits_lower is not None:
        header += f" {'Min':>8} {'Max':>8} {'%Range':>7}"
    lines.append(header)
    lines.append("─" * len(header))

    for i, name in enumerate(names):
        pos = positions[i]
        vel = velocities[i] if velocities is not None else 0.0
        line = f"{name:<12} {pos:>10.4f} {vel:>10.4f}"

        if limits_lower is not None and limits_upper is not None:
            lo, hi = limits_lower[i], limits_upper[i]
            pct = (pos - lo) / (hi - lo) * 100 if hi != lo else 50
            line += f" {lo:>8.3f} {hi:>8.3f} {pct:>6.1f}%"

        lines.append(line)

    return "\n".join(lines)


# =========================================================================
# EE Pose Display
# =========================================================================

def format_ee_pose(pose: np.ndarray) -> str:
    """Format a 4x4 SE(3) pose as position + RPY."""
    pos = pose[:3, 3]
    R = pose[:3, :3]

    import math
    sy = math.sqrt(R[0, 0]**2 + R[1, 0]**2)
    if sy > 1e-6:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0.0

    lines = [
        "End-Effector Pose:",
        f"  Position:    x={pos[0]:.4f}  y={pos[1]:.4f}  z={pos[2]:.4f} m",
        f"  Orientation: r={math.degrees(roll):.1f}°  p={math.degrees(pitch):.1f}°  y={math.degrees(yaw):.1f}°",
    ]
    return "\n".join(lines)
