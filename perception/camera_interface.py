"""
Camera interface abstraction.

Supports both real cameras (via OpenCV/RealSense) and simulated cameras.
All perception modules consume images through this interface.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CameraInfo:
    """Camera intrinsic parameters and metadata."""
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    dist_coeffs: np.ndarray  # distortion coefficients
    frame_id: str = "camera_optical_frame"

    @property
    def camera_matrix(self) -> np.ndarray:
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1],
        ], dtype=np.float64)


class CameraInterface(ABC):
    """Abstract camera interface."""

    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the camera. Returns True on success."""
        ...

    @abstractmethod
    def shutdown(self) -> None:
        """Release camera resources."""
        ...

    @abstractmethod
    def get_frame(self) -> Optional[np.ndarray]:
        """Capture a single BGR frame. Returns None on failure."""
        ...

    @abstractmethod
    def get_camera_info(self) -> CameraInfo:
        """Get camera intrinsic parameters."""
        ...

    def get_depth_frame(self) -> Optional[np.ndarray]:
        """Capture a depth frame (if RGB-D camera). Returns None if unsupported."""
        return None

    def get_frame_undistorted(self) -> Optional[np.ndarray]:
        """Capture and undistort a frame."""
        import cv2
        frame = self.get_frame()
        if frame is None:
            return None
        info = self.get_camera_info()
        return cv2.undistort(frame, info.camera_matrix, info.dist_coeffs)


class OpenCVCamera(CameraInterface):
    """Camera using OpenCV VideoCapture (USB webcams, etc.)."""

    def __init__(self, device_id: int = 0, width: int = 1280, height: int = 720) -> None:
        self._device_id = device_id
        self._width = width
        self._height = height
        self._cap = None
        self._info: Optional[CameraInfo] = None

    def initialize(self) -> bool:
        import cv2
        self._cap = cv2.VideoCapture(self._device_id)
        if not self._cap.isOpened():
            logger.error(f"Failed to open camera {self._device_id}")
            return False

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Default camera info (should be overwritten by calibration)
        self._info = CameraInfo(
            width=actual_w,
            height=actual_h,
            fx=actual_w,  # rough estimate
            fy=actual_w,
            cx=actual_w / 2,
            cy=actual_h / 2,
            dist_coeffs=np.zeros(5),
        )
        logger.info(f"Camera {self._device_id} opened: {actual_w}x{actual_h}")
        return True

    def shutdown(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def get_frame(self) -> Optional[np.ndarray]:
        if self._cap is None:
            return None
        ret, frame = self._cap.read()
        if not ret:
            logger.warning("Failed to capture frame")
            return None
        return frame

    def get_camera_info(self) -> CameraInfo:
        if self._info is None:
            raise RuntimeError("Camera not initialized")
        return self._info

    def set_camera_info(self, info: CameraInfo) -> None:
        """Update camera info (e.g., after calibration)."""
        self._info = info


class SimulatedCamera(CameraInterface):
    """
    Simulated camera for testing the perception pipeline without hardware.

    Generates synthetic board images with colored squares and
    optional piece markers.
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        board_visible: bool = True,
    ) -> None:
        self._width = width
        self._height = height
        self._board_visible = board_visible
        self._frame_count = 0
        self._piece_positions: dict[str, str] = {}  # square → piece_type

    def initialize(self) -> bool:
        logger.info(f"Simulated camera initialized: {self._width}x{self._height}")
        return True

    def shutdown(self) -> None:
        pass

    def get_frame(self) -> Optional[np.ndarray]:
        """Generate a synthetic board image."""
        img = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        img[:] = (40, 40, 40)  # dark background

        if self._board_visible:
            self._draw_board(img)

        self._frame_count += 1
        return img

    def get_camera_info(self) -> CameraInfo:
        return CameraInfo(
            width=self._width,
            height=self._height,
            fx=500.0,
            fy=500.0,
            cx=self._width / 2,
            cy=self._height / 2,
            dist_coeffs=np.zeros(5),
            frame_id="sim_camera_optical_frame",
        )

    def set_piece_positions(self, positions: dict[str, str]) -> None:
        """Set piece positions for rendering. Keys: algebraic squares, values: piece names."""
        self._piece_positions = positions

    def _draw_board(self, img: np.ndarray) -> None:
        """Draw a simple chessboard on the image."""
        import cv2

        # Board region in image (centered, with margin)
        margin = 40
        board_size = min(self._width - 2 * margin, self._height - 2 * margin)
        sq_size = board_size // 8
        x0 = (self._width - 8 * sq_size) // 2
        y0 = (self._height - 8 * sq_size) // 2

        light = (200, 200, 180)  # light squares
        dark = (80, 120, 80)     # dark squares

        for rank in range(8):
            for file in range(8):
                color = light if (file + rank) % 2 == 1 else dark
                x = x0 + file * sq_size
                y = y0 + (7 - rank) * sq_size  # flip: rank 0 at bottom
                cv2.rectangle(img, (x, y), (x + sq_size, y + sq_size), color, -1)

                # Draw piece markers
                sq_name = f"{chr(ord('a') + file)}{rank + 1}"
                if sq_name in self._piece_positions:
                    cx = x + sq_size // 2
                    cy = y + sq_size // 2
                    cv2.circle(img, (cx, cy), sq_size // 4, (0, 0, 255), -1)
