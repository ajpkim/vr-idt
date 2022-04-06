from typing import List

import numpy as np
import pandas as pd

def get_gaze_head_matrices(df: pd.DataFrame, col_name_map: dict) -> (np.array, np.array):
    """Return matrices with 3d vectors for VR world gaze locations and head positions."""
    gaze_cols = [col_name_map[col] for col in col_name_map.keys() if "gaze" in col]
    head_cols = [col_name_map[col] for col in col_name_map.keys() if "head" in col]

    gaze_coords = df.loc[:, gaze_cols].values
    head_coords = df.loc[:, head_cols].values

    return gaze_coords, head_coords

def frequencies(times: pd.Series, time="time") -> pd.Series:
    """Compute the sampling frequency for all data points based on adjacent sample times.

    Args:
        df: pd.DataFrame
        time: Name of column in df which has time data in seconds

    Returns:
        sample_freqs -- pd.Series of sampling rates in hz (samples/sec)
    """
    sample_freqs = times.diff()
    sample_freqs[0] = times.iloc[0]
    sample_freqs = 1 / sample_freqs
    return sample_freqs

def valid_frequences(sample_freqs: np.array, min_freq: float) -> bool:
    """Check that all the frequencies are above given minimum."""
    return all(freq > min_freq for freq in sample_freqs)

def angle_between(v1: np.array, v2: np.array) -> float:
    """Compute the angle theta between vectors v1 and v2.

    The scalar product of v1 and v2 is defined as:
      dot(v1,v2) = mag(v1) * mag(v2) * cos(theta)

    where dot() is a function which computes the dot product and mag()
    is a function which computes the magnitude of the given vector.

    Args:
        v1: vector with dim (m x n)
        v2: with dim (m x n)

    Returns:
        theta: angle between vectors v1 and v2 in degrees.
    """
    norms = np.linalg.norm(v1) * np.linalg.norm(v2)
    cos_theta = np.dot(v1, v2) / norms
    theta = np.arccos(np.clip(cos_theta, -1, 1))
    return np.rad2deg(theta)

def valid_window_angles(window_gaze_coords: np.array, window_head_coords: np.array, max_angle: int) -> bool:
    """Check if angle dispersions within window are valid for fixation classification."""
    vectors = window_gaze_coords - np.mean(window_head_coords, axis=0)
    for i in range(vectors.shape[0]):
        v1 = vectors[i]
        for j in range(i+1, vectors.shape[0]):
            v2 = vectors[j]
            if angle_between(v1, v2) > max_angle:
                return False
    return True

def is_fixation(gaze_coords: np.array,
                head_coords: np.array,
                sample_freqs: np.array,
                window_start: int,
                window_end: int,
                max_angle: float,
                min_freq: int) -> bool:
    """Return a bool indicating whether the given window is part of a fixation."""
    if not valid_frequences(sample_freqs[window_start:window_end+1], min_freq):
        return False
    window_gaze_coords = gaze_coords[window_start:window_end+1]
    window_head_coords = head_coords[window_start:window_end+1]
    return valid_window_angles(window_gaze_coords, window_head_coords, max_angle)

def classify_fixations(df: pd.DataFrame,
                       min_duration: float = 0.15,
                       max_angle: float = 1.50,
                       min_freq: float = 30.0,
                       time: str = "time",
                       gaze_world_x: str = "gaze_world_x",
                       gaze_world_y: str = "gaze_world_y",
                       gaze_world_z: str = "gaze_world_z",
                       head_pos_x: str = "head_pos_x",
                       head_pos_y: str = "head_pos_y",
                       head_pos_z: str = "head_pos_z") -> pd.DataFrame:
    """Classify VR eye fixation events in eye-tracking data.

    Args:
        df: DataFrame with eye tracking data to classify
        min_duration: The minimum length of a fixation in seconds
        max_angle: The maximum angle of dispersion within the fixation within
        min_freq: The minimum required frequency for a fixation to be classified
        time: df column name for time (sec) data
        gaze_world_x: df column name for gaze position in virtual world data
        gaze_world_y: df column name for gaze position in virtual world data
        gaze_world_z: df column name for gaze position in virtual world data
        head_pos_x: df column name for head position in physical space data
        head_pos_y: df column name for head position in physical space data
        head_pos_z: df column name for head position in physical space data
    Returns:
        fixation_df: Copy of original arg 'df' with 4 new fixation related columns:
            "fixation", "fixation_start", "fixation_end", and "fixation_duration"
    """
    # Create mapping to column names in the given df
    col_name_map = {"time": time,
                    "gaze_world_x": gaze_world_x,
                    "gaze_world_y": gaze_world_y,
                    "gaze_world_z": gaze_world_z,
                    "head_pos_x": head_pos_x,
                    "head_pos_y": head_pos_y,
                    "head_pos_z": head_pos_z}

    if not all(col in df.columns for col in list(col_name_map.values())):
        raise Exception(f"DataFrame is missing some columns from <{col_name_map}>")

    # Initialize matrices, results DF, window indices
    sample_freqs = frequencies(df[time])
    gaze_coords, head_coords = get_gaze_head_matrices(df, col_name_map)
    fixation_cols = ["fixation", "fixation_start", "fixation_end", "fixation_duration"]
    fixation_df = pd.DataFrame(np.zeros((df.shape[0], len(fixation_cols)), int), columns=fixation_cols)
    final = df.shape[0] - 1
    window_start = 0

    # Find fixation windows
    while window_start < final:
        window_end = window_start + 1
        # Extend window until total window time exceeds the given minimum valid window time
        while df.loc[window_end, time] - df.loc[window_start, time] < min_duration:
            window_end += 1
            if window_end > final:
                return df.join(fixation_df)

        # Current window isn't a valid fixation, increment start
        if not is_fixation(gaze_coords, head_coords, sample_freqs, window_start, window_end, max_angle, min_freq):
            window_start += 1
        else:
            # Extend the window while in a valid fixation
            while (is_fixation(gaze_coords, head_coords, sample_freqs, window_start, window_end, max_angle, min_freq)
                   and window_end <= final):
                window_end += 1
            window_end -= 1  # decrement since we've exceeded the fixation window in last loop iteration
            # Process the previous fixation window
            duration = df.loc[window_end, time] - df.loc[window_start, time]
            fixation_df.loc[window_start, "fixation_start"] = 1
            fixation_df.loc[window_end, "fixation_end"] = 1
            fixation_df.loc[window_start:window_end+1, "fixation"]= 1
            fixation_df.loc[window_end, "fixation_duration"] = duration

            window_start = window_end

    return df.join(fixation_df)
