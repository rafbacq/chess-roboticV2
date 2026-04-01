"""
DGT e-board adapter.

Reads board state from a DGT electronic chess board via serial/USB.
Implements the BoardStateProvider interface for seamless switching
between DGT and vision-only modes.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import numpy as np

from chess_core.interfaces import PieceColor, PieceType, Square

logger = logging.getLogger(__name__)


class DGTAdapter:
    """
    Interface to DGT electronic chess boards.

    Supports DGT Smart Board, DGT USB Board, and compatible devices.
    Reads piece positions via the DGT protocol over serial connection.

    Usage:
        dgt = DGTAdapter(port="COM3")  # or "/dev/ttyUSB0" on Linux
        dgt.connect()
        state = dgt.get_board_state()
        dgt.disconnect()
    """

    # DGT Protocol constants
    DGT_SEND_BOARD = 0x42
    DGT_SEND_UPDATE = 0x43
    DGT_SEND_BRD = 0x44

    # Piece encoding (DGT protocol)
    PIECE_MAP = {
        0x01: (PieceType.PAWN, PieceColor.WHITE),
        0x02: (PieceType.ROOK, PieceColor.WHITE),
        0x03: (PieceType.KNIGHT, PieceColor.WHITE),
        0x04: (PieceType.BISHOP, PieceColor.WHITE),
        0x05: (PieceType.KING, PieceColor.WHITE),
        0x06: (PieceType.QUEEN, PieceColor.WHITE),
        0x07: (PieceType.PAWN, PieceColor.BLACK),
        0x08: (PieceType.ROOK, PieceColor.BLACK),
        0x09: (PieceType.KNIGHT, PieceColor.BLACK),
        0x0A: (PieceType.BISHOP, PieceColor.BLACK),
        0x0B: (PieceType.KING, PieceColor.BLACK),
        0x0C: (PieceType.QUEEN, PieceColor.BLACK),
    }

    def __init__(
        self,
        port: str = "COM3",
        baudrate: int = 9600,
        flip_board: bool = False,
    ) -> None:
        self._port = port
        self._baudrate = baudrate
        self._flip_board = flip_board
        self._serial = None
        self._connected = False
        self._board_data: dict[str, Optional[tuple[PieceType, PieceColor]]] = {}
        self._lock = threading.Lock()
        self._reader_thread: Optional[threading.Thread] = None
        self._running = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> bool:
        """Connect to the DGT board."""
        try:
            import serial
            self._serial = serial.Serial(
                port=self._port,
                baudrate=self._baudrate,
                timeout=1.0,
            )
            self._connected = True
            logger.info(f"Connected to DGT board on {self._port}")

            # Request initial board state
            self._request_board_state()

            # Start background reader
            self._running = True
            self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)
            self._reader_thread.start()

            return True
        except ImportError:
            logger.error("pyserial not installed. Install with: pip install pyserial")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to DGT board: {e}")
            return False

    def disconnect(self) -> None:
        """Disconnect from the DGT board."""
        self._running = False
        if self._reader_thread:
            self._reader_thread.join(timeout=2.0)
        if self._serial:
            self._serial.close()
        self._connected = False
        logger.info("Disconnected from DGT board")

    def get_board_state(self) -> dict[str, Optional[tuple[PieceType, PieceColor]]]:
        """
        Get current board state from DGT.

        Returns:
            Dict mapping algebraic square names to (PieceType, PieceColor) or None.
        """
        with self._lock:
            return dict(self._board_data)

    def get_occupancy_map(self) -> dict[str, bool]:
        """Get binary occupancy map from DGT state."""
        state = self.get_board_state()
        return {sq: piece is not None for sq, piece in state.items()}

    def _request_board_state(self) -> None:
        """Send a request to the DGT board for full state."""
        if self._serial:
            self._serial.write(bytes([self.DGT_SEND_BOARD]))

    def _read_loop(self) -> None:
        """Background thread: continuously read updates from DGT."""
        while self._running and self._serial:
            try:
                data = self._serial.read(67)  # full board = 67 bytes
                if len(data) >= 67:
                    self._parse_board_data(data)
            except Exception as e:
                if self._running:
                    logger.error(f"DGT read error: {e}")
                    time.sleep(0.5)

    def _parse_board_data(self, data: bytes) -> None:
        """Parse a full board state message from DGT."""
        with self._lock:
            for sq_idx in range(64):
                # DGT sends squares in order: a1, b1, ..., h1, a2, ..., h8
                byte_val = data[3 + sq_idx]  # skip 3-byte header
                file = sq_idx % 8
                rank = sq_idx // 8

                if self._flip_board:
                    file = 7 - file
                    rank = 7 - rank

                sq_name = f"{chr(ord('a') + file)}{rank + 1}"

                if byte_val in self.PIECE_MAP:
                    self._board_data[sq_name] = self.PIECE_MAP[byte_val]
                else:
                    self._board_data[sq_name] = None


class SimulatedDGT:
    """
    Simulated DGT board for testing without hardware.

    Synchronizes with a python-chess Board object to provide
    DGT-like board state reads.
    """

    def __init__(self) -> None:
        self._board_data: dict[str, Optional[tuple[PieceType, PieceColor]]] = {}
        self._connected = True

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> bool:
        return True

    def disconnect(self) -> None:
        self._connected = False

    def sync_with_chess_board(self, board) -> None:
        """Sync state from a python-chess Board object."""
        import chess

        piece_map = {
            chess.PAWN: PieceType.PAWN, chess.KNIGHT: PieceType.KNIGHT,
            chess.BISHOP: PieceType.BISHOP, chess.ROOK: PieceType.ROOK,
            chess.QUEEN: PieceType.QUEEN, chess.KING: PieceType.KING,
        }
        color_map = {chess.WHITE: PieceColor.WHITE, chess.BLACK: PieceColor.BLACK}

        self._board_data.clear()
        for sq in chess.SQUARES:
            file = chess.square_file(sq)
            rank = chess.square_rank(sq)
            sq_name = f"{chr(ord('a') + file)}{rank + 1}"
            piece = board.piece_at(sq)
            if piece:
                self._board_data[sq_name] = (piece_map[piece.piece_type], color_map[piece.color])
            else:
                self._board_data[sq_name] = None

    def get_board_state(self) -> dict[str, Optional[tuple[PieceType, PieceColor]]]:
        return dict(self._board_data)

    def get_occupancy_map(self) -> dict[str, bool]:
        return {sq: piece is not None for sq, piece in self._board_data.items()}
