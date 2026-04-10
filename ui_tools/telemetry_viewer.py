"""
Telemetry replay and analysis tool.

Loads saved telemetry .npz files from move executions and provides
analysis utilities: trajectory plotting data, velocity profiles,
timing breakdowns, and statistical summaries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryAnalysis:
    """Analysis of a single execution trajectory."""
    move_uci: str
    total_duration_s: float
    n_samples: int
    max_joint_velocity_rads: float
    mean_joint_velocity_rads: float
    max_ee_velocity_ms: float
    path_length_m: float
    joint_range_used_rad: np.ndarray  # per-joint range used
    gripper_transitions: int  # number of open/close transitions


def load_telemetry(filepath: str | Path) -> dict:
    """
    Load a telemetry .npz file.

    Returns:
        Dict with keys: timestamps, joint_positions, ee_poses, gripper_width_mm.
    """
    data = np.load(str(filepath))
    return {
        "timestamps": data["timestamps"],
        "joint_positions": data["joint_positions"],
        "ee_poses": data["ee_poses"],
        "gripper_width_mm": data["gripper_width_mm"],
    }


def analyze_trajectory(telemetry: dict, move_uci: str = "") -> TrajectoryAnalysis:
    """
    Analyze a loaded telemetry trajectory.

    Args:
        telemetry: Dict from load_telemetry().
        move_uci: Optional move label.

    Returns:
        TrajectoryAnalysis with computed metrics.
    """
    ts = telemetry["timestamps"]
    joints = telemetry["joint_positions"]
    ee_poses = telemetry["ee_poses"]
    gripper = telemetry["gripper_width_mm"]

    n = len(ts)
    if n < 2:
        return TrajectoryAnalysis(
            move_uci=move_uci, total_duration_s=0, n_samples=n,
            max_joint_velocity_rads=0, mean_joint_velocity_rads=0,
            max_ee_velocity_ms=0, path_length_m=0,
            joint_range_used_rad=np.zeros(joints.shape[1] if len(joints.shape) > 1 else 6),
            gripper_transitions=0,
        )

    total_duration = ts[-1] - ts[0]
    dt = np.diff(ts)
    dt = np.where(dt > 0, dt, 1e-6)  # avoid division by zero

    # Joint velocities
    joint_diffs = np.diff(joints, axis=0)
    joint_vels = joint_diffs / dt[:, np.newaxis]
    max_jvel = float(np.max(np.abs(joint_vels)))
    mean_jvel = float(np.mean(np.abs(joint_vels)))

    # EE velocities and path length
    ee_positions = ee_poses[:, :3, 3] if ee_poses.ndim == 3 else ee_poses[:, :3]
    ee_diffs = np.diff(ee_positions, axis=0)
    ee_dists = np.linalg.norm(ee_diffs, axis=1)
    ee_vels = ee_dists / dt
    max_ee_vel = float(np.max(ee_vels))
    path_length = float(np.sum(ee_dists))

    # Joint range
    joint_range = np.ptp(joints, axis=0)

    # Gripper transitions
    gripper_binary = (gripper > 10).astype(int)  # open > 10mm
    transitions = int(np.sum(np.abs(np.diff(gripper_binary))))

    return TrajectoryAnalysis(
        move_uci=move_uci,
        total_duration_s=total_duration,
        n_samples=n,
        max_joint_velocity_rads=max_jvel,
        mean_joint_velocity_rads=mean_jvel,
        max_ee_velocity_ms=max_ee_vel,
        path_length_m=path_length,
        joint_range_used_rad=joint_range,
        gripper_transitions=transitions,
    )


def format_analysis(analysis: TrajectoryAnalysis) -> str:
    """Format a trajectory analysis as a readable string."""
    lines = [
        f"Trajectory Analysis: {analysis.move_uci or '(unnamed)'}",
        f"  Duration:        {analysis.total_duration_s:.3f}s",
        f"  Samples:         {analysis.n_samples}",
        f"  Path length:     {analysis.path_length_m * 1000:.1f} mm",
        f"  Max joint vel:   {analysis.max_joint_velocity_rads:.3f} rad/s",
        f"  Mean joint vel:  {analysis.mean_joint_velocity_rads:.3f} rad/s",
        f"  Max EE vel:      {analysis.max_ee_velocity_ms * 1000:.1f} mm/s",
        f"  Gripper changes: {analysis.gripper_transitions}",
    ]
    return "\n".join(lines)


def batch_analyze(telemetry_dir: str | Path) -> list[TrajectoryAnalysis]:
    """Analyze all telemetry files in a directory."""
    path = Path(telemetry_dir)
    results = []
    for f in sorted(path.glob("telemetry_*.npz")):
        try:
            data = load_telemetry(f)
            # Extract move UCI from filename: telemetry_{uci}_{timestamp}.npz
            name_parts = f.stem.split("_")
            uci = name_parts[1] if len(name_parts) >= 2 else ""
            results.append(analyze_trajectory(data, uci))
        except Exception as e:
            logger.warning(f"Failed to analyze {f}: {e}")
    return results


def print_batch_summary(analyses: list[TrajectoryAnalysis]) -> None:
    """Print a summary table of multiple trajectory analyses."""
    if not analyses:
        print("No trajectories to analyze.")
        return

    print(f"\n{'Move':<8} {'Time':>6} {'Path(mm)':>9} {'MaxJVel':>8} {'MaxEEVel':>9} {'GripChg':>8}")
    print("─" * 56)
    for a in analyses:
        print(
            f"{a.move_uci:<8} {a.total_duration_s:>5.2f}s "
            f"{a.path_length_m * 1000:>8.1f} {a.max_joint_velocity_rads:>8.3f} "
            f"{a.max_ee_velocity_ms * 1000:>8.1f} {a.gripper_transitions:>8}"
        )

    durations = [a.total_duration_s for a in analyses]
    paths = [a.path_length_m for a in analyses]
    print("─" * 56)
    print(f"{'AVG':<8} {np.mean(durations):>5.2f}s {np.mean(paths) * 1000:>8.1f}")
    print(f"{'TOTAL':<8} {sum(durations):>5.2f}s {sum(paths) * 1000:>8.1f}")
