"""
Utility functions for PGC dataset.
"""

import ast
import re
from typing import Tuple

import numpy as np


def parse_positions(positions_str, num_players: int = 4) -> np.ndarray:
    """
    Parse positions string to numpy array.

    Args:
        positions_str: String representation of positions list.
            Example: "[[189090, 417635.90625, 7173.60986328125], ...]"
            May contain 'nan' values for dead players.
            Can also be NaN (float) if data is missing.
        num_players: Expected number of players per squad. Default is 4.

    Returns:
        numpy array of shape (num_players, 3) containing [x, y, z] for each player.
        Dead players (nan or [0, 0, 0] coordinates) are set to nan.
        If fewer players exist, remaining slots are filled with nan.
    """
    # Handle NaN or non-string input
    if not isinstance(positions_str, str):
        # Return all NaN positions for invalid input
        return np.full((num_players, 3), np.nan, dtype=np.float32)
    
    # Replace 'nan' string with 'None' so ast.literal_eval can parse it
    positions_str_clean = positions_str.replace('nan', 'None')
    positions_list = ast.literal_eval(positions_str_clean)

    # Initialize with nan (for missing players)
    positions = np.full((num_players, 3), np.nan, dtype=np.float32)

    # Fill in available positions
    actual_players = min(len(positions_list), num_players)
    for i in range(actual_players):
        player_pos = positions_list[i]
        # Check if any coordinate is None (was nan)
        if player_pos is None or any(p is None for p in player_pos):
            # Keep as nan (already initialized)
            continue
        positions[i] = np.array(player_pos, dtype=np.float32)

    # Replace [0, 0, 0] with nan (dead players)
    dead_mask = np.all(positions == 0.0, axis=1)
    positions[dead_mask] = np.nan

    return positions


def parse_zone_info(zone_str) -> np.ndarray:
    """
    Parse bluezone_info or whitezone_info string to numpy array.

    Args:
        zone_str: String representation of zone tuple.
            Example: "(np.float64(406387.5), np.float64(406387.5),
                       np.float64(579718.6875))"
            Can also be NaN (float) if data is missing.

    Returns:
        numpy array of shape (3,) containing [x, y, z].
    """
    # Handle NaN or non-string input
    if not isinstance(zone_str, str):
        return np.full(3, np.nan, dtype=np.float32)
    
    # Extract numbers from np.float64(...) pattern
    pattern = r'np\.float64\(([-\d.]+)\)'
    matches = re.findall(pattern, zone_str)

    if len(matches) == 3:
        return np.array([float(m) for m in matches], dtype=np.float32)

    # Fallback: try ast.literal_eval for simple tuple format
    zone_str_clean = zone_str.replace('np.float64', '')
    values = ast.literal_eval(zone_str_clean)
    return np.array(values, dtype=np.float32)


def parse_location_features(
    positions_str: str,
    bluezone_str: str,
    whitezone_str: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse all location features.

    Args:
        positions_str: String representation of positions.
        bluezone_str: String representation of bluezone_info.
        whitezone_str: String representation of whitezone_info.

    Returns:
        Tuple of (positions, bluezone_info, whitezone_info) as numpy arrays.
            - positions: shape (4, 3)
            - bluezone_info: shape (3,)
            - whitezone_info: shape (3,)
    """
    positions = parse_positions(positions_str)
    bluezone_info = parse_zone_info(bluezone_str)
    whitezone_info = parse_zone_info(whitezone_str)

    return positions, bluezone_info, whitezone_info


# In-game phase boundaries (seconds). Phase k covers
# [PHASE_BOUNDARIES[k-1], PHASE_BOUNDARIES[k]).
PHASE_BOUNDARIES = [0, 600, 810, 990, 1170, 1350, 1530, 1680, 1800, 1970, 2000]
NUM_PHASES = len(PHASE_BOUNDARIES) - 1


def time_point_to_phase(time_point: float) -> int:
    """
    Map an in-game time point (seconds) to its phase number (1-10).
    """
    for phase in range(1, NUM_PHASES + 1):
        if time_point < PHASE_BOUNDARIES[phase]:
            return phase
    return NUM_PHASES
