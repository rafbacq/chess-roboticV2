"""
Transform manager and calibration persistence.

Manages the graph of coordinate frame transforms:
    world/robot_base → camera → board → squares → EE → grasp

Provides utilities for:
    - Transform composition and inversion
    - Transform consistency checking
    - Serialization to/from YAML
    - Reprojection error computation
    - Visual diagnostics
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from chess_core.interfaces import CalibrationBundle

logger = logging.getLogger(__name__)


class TransformManager:
    """
    Manages named SE(3) transforms between coordinate frames.

    Usage:
        tm = TransformManager()
        tm.set_transform("robot_base", "camera", T_robot_camera)
        tm.set_transform("camera", "board", T_camera_board)

        # Get composed transform
        T_robot_board = tm.get_transform("robot_base", "board")

        # Check consistency
        errors = tm.check_consistency()
    """

    def __init__(self) -> None:
        # Stores transforms as T_parent_child (transform from child to parent)
        self._transforms: dict[tuple[str, str], np.ndarray] = {}
        self._frames: set[str] = set()

    def set_transform(
        self,
        parent_frame: str,
        child_frame: str,
        transform: np.ndarray,
    ) -> None:
        """
        Set the transform from child_frame to parent_frame.

        Convention: T_parent_child means points in child frame
        can be transformed to parent frame via: p_parent = T @ p_child

        Args:
            parent_frame: Name of the parent frame.
            child_frame: Name of the child frame.
            transform: 4x4 SE(3) homogeneous transformation matrix.
        """
        transform = np.asarray(transform, dtype=np.float64)
        assert transform.shape == (4, 4), f"Transform must be 4x4, got {transform.shape}"

        self._transforms[(parent_frame, child_frame)] = transform.copy()
        # Also store the inverse
        self._transforms[(child_frame, parent_frame)] = np.linalg.inv(transform)

        self._frames.add(parent_frame)
        self._frames.add(child_frame)

        logger.debug(f"Set transform: {parent_frame} ← {child_frame}")

    def get_transform(self, target_frame: str, source_frame: str) -> np.ndarray:
        """
        Get the transform from source_frame to target_frame.

        If no direct transform exists, attempts to compose through
        available intermediate frames (BFS).

        Returns:
            4x4 SE(3) transform.

        Raises:
            KeyError: If no path exists between frames.
        """
        if target_frame == source_frame:
            return np.eye(4, dtype=np.float64)

        # Direct lookup
        key = (target_frame, source_frame)
        if key in self._transforms:
            return self._transforms[key].copy()

        # BFS for path
        path = self._find_path(target_frame, source_frame)
        if path is None:
            raise KeyError(
                f"No transform path from '{source_frame}' to '{target_frame}'. "
                f"Available frames: {self._frames}"
            )

        # Compose transforms along path
        T = np.eye(4, dtype=np.float64)
        for i in range(len(path) - 1):
            T = self._transforms[(path[i], path[i + 1])] @ T

        return T

    def transform_point(
        self,
        point: np.ndarray,
        target_frame: str,
        source_frame: str,
    ) -> np.ndarray:
        """Transform a 3D point from source_frame to target_frame."""
        T = self.get_transform(target_frame, source_frame)
        p_hom = np.ones(4)
        p_hom[:3] = point[:3]
        return (T @ p_hom)[:3]

    def transform_points(
        self,
        points: np.ndarray,
        target_frame: str,
        source_frame: str,
    ) -> np.ndarray:
        """Transform an array of 3D points (Nx3) from source to target frame."""
        T = self.get_transform(target_frame, source_frame)
        N = points.shape[0]
        p_hom = np.hstack([points, np.ones((N, 1))])
        return (T @ p_hom.T).T[:, :3]

    def _find_path(self, start: str, end: str) -> Optional[list[str]]:
        """BFS to find a path between frames."""
        if start not in self._frames or end not in self._frames:
            return None

        # Build adjacency from stored transforms
        adjacency: dict[str, set[str]] = {f: set() for f in self._frames}
        for (parent, child) in self._transforms:
            adjacency[parent].add(child)

        visited = {start}
        queue = [[start]]

        while queue:
            path = queue.pop(0)
            current = path[-1]

            if current == end:
                return path

            for neighbor in adjacency.get(current, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(path + [neighbor])

        return None

    def check_consistency(self, tolerance: float = 1e-6) -> list[str]:
        """
        Check transform consistency (inverse correctness, loop closure).

        Returns:
            List of warning messages. Empty = all consistent.
        """
        warnings = []

        for (parent, child), T in self._transforms.items():
            # Check inverse consistency
            inv_key = (child, parent)
            if inv_key in self._transforms:
                T_inv = self._transforms[inv_key]
                product = T @ T_inv
                identity_err = np.linalg.norm(product - np.eye(4))
                if identity_err > tolerance:
                    warnings.append(
                        f"Inverse inconsistency: {parent}↔{child}, "
                        f"error={identity_err:.2e}"
                    )

            # Check rotation matrix validity
            R = T[:3, :3]
            det = np.linalg.det(R)
            if abs(det - 1.0) > tolerance:
                warnings.append(
                    f"Rotation not SO(3): {parent}←{child}, det={det:.6f}"
                )

            orthogonality = np.linalg.norm(R @ R.T - np.eye(3))
            if orthogonality > tolerance:
                warnings.append(
                    f"Rotation not orthogonal: {parent}←{child}, "
                    f"error={orthogonality:.2e}"
                )

        if warnings:
            for w in warnings:
                logger.warning(f"Transform consistency: {w}")
        else:
            logger.debug("All transforms consistent")

        return warnings

    @property
    def frames(self) -> set[str]:
        return self._frames.copy()

    def save(self, filepath: str) -> None:
        """Save all transforms to a YAML file."""
        data = {}
        seen = set()

        for (parent, child), T in self._transforms.items():
            key = tuple(sorted([parent, child]))
            if key in seen:
                continue
            seen.add(key)

            data[f"{parent}_from_{child}"] = {
                "parent": parent,
                "child": child,
                "transform": T.tolist(),
            }

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
        logger.info(f"Transforms saved to {filepath}")

    def load(self, filepath: str) -> None:
        """Load transforms from a YAML file."""
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)

        for name, entry in data.items():
            T = np.array(entry["transform"], dtype=np.float64)
            self.set_transform(entry["parent"], entry["child"], T)

        logger.info(f"Loaded {len(data)} transforms from {filepath}")


def save_calibration(bundle: CalibrationBundle, filepath: str) -> None:
    """Save a CalibrationBundle to a JSON file."""
    data = {
        "camera_matrix": bundle.camera_matrix.tolist(),
        "dist_coeffs": bundle.dist_coeffs.tolist(),
        "T_camera_board": bundle.T_camera_board.tolist(),
        "T_robot_board": bundle.T_robot_board.tolist(),
        "T_robot_camera": bundle.T_robot_camera.tolist(),
        "reprojection_error_px": bundle.reprojection_error_px,
        "timestamp": bundle.timestamp,
        "valid": bundle.valid,
        "notes": bundle.notes,
    }

    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    logger.info(f"Calibration saved to {filepath}")


def load_calibration(filepath: str) -> CalibrationBundle:
    """Load a CalibrationBundle from a JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    bundle = CalibrationBundle(
        camera_matrix=np.array(data["camera_matrix"]),
        dist_coeffs=np.array(data["dist_coeffs"]),
        T_camera_board=np.array(data["T_camera_board"]),
        T_robot_board=np.array(data["T_robot_board"]),
        T_robot_camera=np.array(data["T_robot_camera"]),
        reprojection_error_px=data.get("reprojection_error_px", 0.0),
        timestamp=data.get("timestamp", 0.0),
        valid=data.get("valid", True),
        notes=data.get("notes", ""),
    )

    logger.info(
        f"Calibration loaded from {filepath} "
        f"(reprojection error: {bundle.reprojection_error_px:.3f}px)"
    )
    return bundle
