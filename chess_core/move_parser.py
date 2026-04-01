"""
Move parser: converts between UCI strings, algebraic notation, and internal
ChessMove representations with full move-type classification.

Uses python-chess as the underlying chess library for correctness.
"""

from __future__ import annotations

import logging
from typing import Optional

import chess

from chess_core.interfaces import (
    ChessMove,
    MoveType,
    PieceColor,
    PieceType,
    Square,
)

logger = logging.getLogger(__name__)

# Mapping from python-chess piece types to our PieceType enum
_PIECE_MAP: dict[int, PieceType] = {
    chess.PAWN: PieceType.PAWN,
    chess.KNIGHT: PieceType.KNIGHT,
    chess.BISHOP: PieceType.BISHOP,
    chess.ROOK: PieceType.ROOK,
    chess.QUEEN: PieceType.QUEEN,
    chess.KING: PieceType.KING,
}

_COLOR_MAP: dict[bool, PieceColor] = {
    chess.WHITE: PieceColor.WHITE,
    chess.BLACK: PieceColor.BLACK,
}

# Reverse map for promotion specification
_PROMOTION_MAP: dict[PieceType, int] = {
    PieceType.QUEEN: chess.QUEEN,
    PieceType.ROOK: chess.ROOK,
    PieceType.BISHOP: chess.BISHOP,
    PieceType.KNIGHT: chess.KNIGHT,
}


def parse_uci_move(uci_string: str, board: chess.Board) -> ChessMove:
    """
    Parse a UCI move string (e.g. "e2e4", "e7e8q") into a ChessMove
    with full move-type classification.

    Args:
        uci_string: UCI move string.
        board: Current board position (needed to classify move type).

    Returns:
        Fully classified ChessMove.

    Raises:
        ValueError: If the move is illegal or the UCI string is invalid.
    """
    try:
        move = chess.Move.from_uci(uci_string)
    except (ValueError, chess.InvalidMoveError) as e:
        raise ValueError(f"Invalid UCI move string '{uci_string}': {e}")

    if move not in board.legal_moves:
        raise ValueError(
            f"Move '{uci_string}' is not legal in position: {board.fen()}"
        )

    # Source and target squares
    source = Square(file=chess.square_file(move.from_square),
                    rank=chess.square_rank(move.from_square))
    target = Square(file=chess.square_file(move.to_square),
                    rank=chess.square_rank(move.to_square))

    # Piece being moved
    piece_at_src = board.piece_at(move.from_square)
    if piece_at_src is None:
        raise ValueError(f"No piece at source square {source.algebraic}")

    piece_type = _PIECE_MAP[piece_at_src.piece_type]
    color = _COLOR_MAP[piece_at_src.color]

    # Classify move type
    move_type = _classify_move(board, move, piece_type)

    # Captured piece (if any)
    captured_piece: Optional[PieceType] = None
    if board.is_capture(move):
        if board.is_en_passant(move):
            captured_piece = PieceType.PAWN
        else:
            captured = board.piece_at(move.to_square)
            if captured:
                captured_piece = _PIECE_MAP[captured.piece_type]

    # Promotion piece (if any)
    promotion_piece: Optional[PieceType] = None
    if move.promotion is not None:
        promotion_piece = _PIECE_MAP[move.promotion]

    result = ChessMove(
        source=source,
        target=target,
        move_type=move_type,
        piece=piece_type,
        color=color,
        captured_piece=captured_piece,
        promotion_piece=promotion_piece,
        uci_string=uci_string,
    )

    logger.debug(f"Parsed UCI '{uci_string}' → {result}")
    return result


def _classify_move(board: chess.Board, move: chess.Move, piece: PieceType) -> MoveType:
    """Classify a move by its physical execution requirements."""
    is_capture = board.is_capture(move)
    is_en_passant = board.is_en_passant(move)
    is_castling = board.is_castling(move)
    is_promotion = move.promotion is not None

    if is_en_passant:
        return MoveType.EN_PASSANT
    if is_castling:
        if chess.square_file(move.to_square) > chess.square_file(move.from_square):
            return MoveType.CASTLING_KINGSIDE
        else:
            return MoveType.CASTLING_QUEENSIDE
    if is_promotion and is_capture:
        return MoveType.PROMOTION_CAPTURE
    if is_promotion:
        return MoveType.PROMOTION
    if is_capture:
        return MoveType.CAPTURE
    return MoveType.NORMAL


def chess_move_to_uci(move: ChessMove) -> str:
    """Convert a ChessMove back to UCI string."""
    uci = f"{move.source.algebraic}{move.target.algebraic}"
    if move.promotion_piece is not None:
        # UCI promotion uses lowercase piece letter
        promo_chars = {
            PieceType.QUEEN: 'q',
            PieceType.ROOK: 'r',
            PieceType.BISHOP: 'b',
            PieceType.KNIGHT: 'n',
        }
        uci += promo_chars.get(move.promotion_piece, 'q')
    return uci


def get_castling_rook_move(move: ChessMove) -> tuple[Square, Square]:
    """
    For a castling move, return the rook's source and target squares.

    Args:
        move: A castling ChessMove.

    Returns:
        (rook_source, rook_target) squares.

    Raises:
        ValueError: If the move is not a castling move.
    """
    if not move.is_castling:
        raise ValueError(f"Not a castling move: {move}")

    rank = move.source.rank  # same rank for king and rook

    if move.move_type == MoveType.CASTLING_KINGSIDE:
        # King: e→g, Rook: h→f
        rook_src = Square(file=7, rank=rank)
        rook_tgt = Square(file=5, rank=rank)
    else:
        # King: e→c, Rook: a→d
        rook_src = Square(file=0, rank=rank)
        rook_tgt = Square(file=3, rank=rank)

    return rook_src, rook_tgt


def get_en_passant_capture_square(move: ChessMove) -> Square:
    """
    For an en passant move, return the square where the captured pawn actually sits.

    The captured pawn is on the same file as the target but the same rank as the source.

    Args:
        move: An en passant ChessMove.

    Returns:
        Square where the captured pawn is.

    Raises:
        ValueError: If the move is not en passant.
    """
    if move.move_type != MoveType.EN_PASSANT:
        raise ValueError(f"Not an en passant move: {move}")

    return Square(file=move.target.file, rank=move.source.rank)
