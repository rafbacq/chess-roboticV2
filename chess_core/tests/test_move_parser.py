"""
Unit tests for Square, move parsing, and algebraic notation conversion.
"""

import pytest
import chess

from chess_core.interfaces import (
    ChessMove,
    MoveType,
    PieceColor,
    PieceType,
    Square,
    SquarePose,
)
from chess_core.move_parser import (
    chess_move_to_uci,
    get_castling_rook_move,
    get_en_passant_capture_square,
    parse_uci_move,
)


class TestSquare:
    """Tests for the Square data model."""

    def test_from_algebraic_a1(self):
        sq = Square.from_algebraic("a1")
        assert sq.file == 0
        assert sq.rank == 0

    def test_from_algebraic_h8(self):
        sq = Square.from_algebraic("h8")
        assert sq.file == 7
        assert sq.rank == 7

    def test_from_algebraic_e4(self):
        sq = Square.from_algebraic("e4")
        assert sq.file == 4
        assert sq.rank == 3

    def test_algebraic_roundtrip(self):
        """Every square should survive algebraic→Square→algebraic."""
        for f in range(8):
            for r in range(8):
                sq = Square(file=f, rank=r)
                alg = sq.algebraic
                sq2 = Square.from_algebraic(alg)
                assert sq == sq2, f"Roundtrip failed for {sq}"

    def test_invalid_algebraic_too_long(self):
        with pytest.raises(ValueError):
            Square.from_algebraic("e44")

    def test_invalid_algebraic_bad_file(self):
        with pytest.raises(ValueError):
            Square.from_algebraic("z4")

    def test_invalid_algebraic_bad_rank(self):
        with pytest.raises(ValueError):
            Square.from_algebraic("a9")

    def test_invalid_file_range(self):
        with pytest.raises(ValueError):
            Square(file=8, rank=0)

    def test_invalid_rank_range(self):
        with pytest.raises(ValueError):
            Square(file=0, rank=-1)

    def test_light_dark_squares(self):
        # a1 is dark (file=0, rank=0 → sum=0 → even → dark)
        assert not Square.from_algebraic("a1").is_light_square
        # b1 is light
        assert Square.from_algebraic("b1").is_light_square
        # h8 is dark
        assert not Square.from_algebraic("h8").is_light_square

    def test_frozen(self):
        """Square is immutable."""
        sq = Square(file=4, rank=3)
        with pytest.raises(AttributeError):
            sq.file = 5


class TestMoveParser:
    """Tests for UCI move parsing and classification."""

    def test_simple_pawn_advance(self):
        board = chess.Board()
        move = parse_uci_move("e2e4", board)
        assert move.source == Square.from_algebraic("e2")
        assert move.target == Square.from_algebraic("e4")
        assert move.piece == PieceType.PAWN
        assert move.color == PieceColor.WHITE
        assert move.move_type == MoveType.NORMAL
        assert not move.is_capture
        assert not move.is_castling

    def test_knight_move(self):
        board = chess.Board()
        move = parse_uci_move("g1f3", board)
        assert move.piece == PieceType.KNIGHT
        assert move.move_type == MoveType.NORMAL

    def test_capture(self):
        # Setup a position where e4 pawn can capture d5 pawn
        board = chess.Board("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2")
        move = parse_uci_move("e4d5", board)
        assert move.move_type == MoveType.CAPTURE
        assert move.captured_piece == PieceType.PAWN
        assert move.is_capture

    def test_kingside_castling(self):
        board = chess.Board("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1")
        move = parse_uci_move("e1g1", board)
        assert move.move_type == MoveType.CASTLING_KINGSIDE
        assert move.piece == PieceType.KING
        assert move.is_castling

    def test_queenside_castling(self):
        board = chess.Board("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1")
        move = parse_uci_move("e1c1", board)
        assert move.move_type == MoveType.CASTLING_QUEENSIDE
        assert move.is_castling

    def test_en_passant(self):
        # White pawn on e5, black just played d7d5
        board = chess.Board("rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3")
        move = parse_uci_move("e5d6", board)
        assert move.move_type == MoveType.EN_PASSANT
        assert move.captured_piece == PieceType.PAWN

    def test_promotion(self):
        board = chess.Board("8/4P3/8/8/8/8/8/4K2k w - - 0 1")
        move = parse_uci_move("e7e8q", board)
        assert move.move_type == MoveType.PROMOTION
        assert move.promotion_piece == PieceType.QUEEN
        assert move.is_promotion

    def test_promotion_capture(self):
        board = chess.Board("3r4/4P3/8/8/8/8/8/4K2k w - - 0 1")
        move = parse_uci_move("e7d8q", board)
        assert move.move_type == MoveType.PROMOTION_CAPTURE
        assert move.captured_piece == PieceType.ROOK
        assert move.promotion_piece == PieceType.QUEEN

    def test_illegal_move_raises(self):
        board = chess.Board()
        with pytest.raises(ValueError, match="not legal"):
            parse_uci_move("e2e5", board)  # can't move 3 squares

    def test_invalid_uci_raises(self):
        board = chess.Board()
        with pytest.raises(ValueError):
            parse_uci_move("xyz", board)

    def test_uci_roundtrip(self):
        board = chess.Board()
        move = parse_uci_move("e2e4", board)
        uci = chess_move_to_uci(move)
        assert uci == "e2e4"

    def test_uci_promotion_roundtrip(self):
        board = chess.Board("8/4P3/8/8/8/8/8/4K2k w - - 0 1")
        move = parse_uci_move("e7e8q", board)
        uci = chess_move_to_uci(move)
        assert uci == "e7e8q"


class TestCastlingRookMove:
    def test_white_kingside(self):
        move = ChessMove(
            source=Square.from_algebraic("e1"),
            target=Square.from_algebraic("g1"),
            move_type=MoveType.CASTLING_KINGSIDE,
            piece=PieceType.KING,
            color=PieceColor.WHITE,
            uci_string="e1g1",
        )
        rook_src, rook_tgt = get_castling_rook_move(move)
        assert rook_src == Square.from_algebraic("h1")
        assert rook_tgt == Square.from_algebraic("f1")

    def test_white_queenside(self):
        move = ChessMove(
            source=Square.from_algebraic("e1"),
            target=Square.from_algebraic("c1"),
            move_type=MoveType.CASTLING_QUEENSIDE,
            piece=PieceType.KING,
            color=PieceColor.WHITE,
            uci_string="e1c1",
        )
        rook_src, rook_tgt = get_castling_rook_move(move)
        assert rook_src == Square.from_algebraic("a1")
        assert rook_tgt == Square.from_algebraic("d1")

    def test_black_kingside(self):
        move = ChessMove(
            source=Square.from_algebraic("e8"),
            target=Square.from_algebraic("g8"),
            move_type=MoveType.CASTLING_KINGSIDE,
            piece=PieceType.KING,
            color=PieceColor.BLACK,
            uci_string="e8g8",
        )
        rook_src, rook_tgt = get_castling_rook_move(move)
        assert rook_src == Square.from_algebraic("h8")
        assert rook_tgt == Square.from_algebraic("f8")

    def test_non_castling_raises(self):
        move = ChessMove(
            source=Square.from_algebraic("e2"),
            target=Square.from_algebraic("e4"),
            move_type=MoveType.NORMAL,
            piece=PieceType.PAWN,
            color=PieceColor.WHITE,
        )
        with pytest.raises(ValueError):
            get_castling_rook_move(move)


class TestEnPassantCapture:
    def test_white_en_passant(self):
        move = ChessMove(
            source=Square.from_algebraic("e5"),
            target=Square.from_algebraic("d6"),
            move_type=MoveType.EN_PASSANT,
            piece=PieceType.PAWN,
            color=PieceColor.WHITE,
            captured_piece=PieceType.PAWN,
        )
        cap_sq = get_en_passant_capture_square(move)
        # Captured pawn is on d5 (same file as target, same rank as source)
        assert cap_sq == Square.from_algebraic("d5")

    def test_black_en_passant(self):
        move = ChessMove(
            source=Square.from_algebraic("d4"),
            target=Square.from_algebraic("e3"),
            move_type=MoveType.EN_PASSANT,
            piece=PieceType.PAWN,
            color=PieceColor.BLACK,
            captured_piece=PieceType.PAWN,
        )
        cap_sq = get_en_passant_capture_square(move)
        assert cap_sq == Square.from_algebraic("e4")
