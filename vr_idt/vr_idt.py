import argparse
import math
import os

from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
# Suppress pandas warning that we can safely ignore
# pd.options.mode.chained_assignment = None

"""
In the process of refactoring this.
"""

GAZE_COLS = ["gaze_world_z", "gaze_world_y", "gaze_world_z"]
HEAD_COLS = ["head_pos_z", "head_pos_y", "head_pos_z"]


def get_gaze_head_matrices(df, cols):
    """TODO"""
    gaze_cols = [cols[x] for x in GAZE_COLS]
    head_cols = [cols[x] for x in HEAD_COLS]

    gaze_matrix = df.loc[:, [gaze_cols]].values
    head_matrix = df.loc[:, [head_cols]].values

    return gaze_matrix, head_matrix



def frequencies(df, time="time") -> pd.Series:
    """Compute the sampling frequency for all data points based on
    adjacent sample times.

    Args:
    df --  pd.DataFrame
    time -- Name of columns in df which has time data in seconds

    Returns:
    sample_freq -- pd.Series of sampling rates
    """
    sample_freq = df["time"].diff()
    sample_freq[0] = df["time"].iloc[0]
    return sample_freq

def angle_between(v1: np.array, v2: np.array) -> float:
    """Compute the angle theta between vectors v1 and v2.

    The scalar product of v1 and v2 is defined as:

      dot(v1,v2) = mag(v1) * mag(v2) * cos(theta)

    where dot() is a function which computes the dot product and mag()
    is a function which computes the magnitude of the given vector.

    Args:
    v1 -- vector with dim (m x n)
    v2 -- vector with dim (m x n)

    Returns:
    theta -- the angle between vectors v1 and v2 in degrees.
    """
    norms = np.linalg.norm(v1) * np.linalg.norm(v2)
    cos_theta = np.dot(v1, v2) / norms
    theta = np.arccos(np.clip(cos_theta, -1, 1))
    return np.rad2deg(theta)


def valid_window_angles(gaze_world_matrix, head_pos_matrix,  max_angle):
    """Check whether the dispersion angles within the given window data
    are within the valid fixation threshold defined by max_angle.
    """
    vectors = gaze_world_matrix - np.mean(head_pos_matrix, axis=0)
    for i in range(vectors.shape[0]):
        v1 = vectors[i]
        for j in range(i+1, vectors.shape[0]):
            v2 = vectors[j]
            if angle_between(v1, v2) > max_angle:
                return False
    return True

def valid_frequences(hz: np.array, min_freq: float) -> bool:
    """TODO"""
    return any(freq < min_freq for freq in hz]


def fixating(df,
             window_start,
             window_end,
             dispersion_threshold=1.5,
             min_freq=30) -> bool:
    """Returns a bool indicating whether the given window is part of a fixation"""
    window_data = df.iloc[window_start:window_end+1]
    if not all(hz > min_freq for hz in window_data['hz']):
        return False

    return check_window_dispersions(window_data, dispersion_threshold)


def vr_itd(df: pd.DataFrame,
           min_window_time      =0.15,
           dispersion_threshold =1.50,
           min_freq             =30,
           time                 ="time",
           gaze_world_x         ="head_pos_z",
           gaze_world_y         ="head_pos_z",
           gaze_world_z         ="head_pos_z",
           head_pos_x           ="head_pos_z",
           head_pos_y           ="head_pos_z",
           head_pos_z           ="head_pos_z",
           inplace              =False,
           ) -> pd.DataFrame:
    """
    Implements the VR IDT algorithm as proposed in:
    https://www.mdpi.com/1424-8220/20/17/4956. Original code from
    authors is here:
    https://github.com/ASAPLableni/VR-centred_I-DT_algorithm.
    """

    # Mapping to column names in the given df
    cols = {"time": time,
            "gaze_world_x": gaze_world_x,
            "gaze_world_y": gaze_world_y,
            "gaze_world_z": gaze_world_z,
            "head_pos_x": head_pos_x,
            "head_pos_y": head_pos_y,
            "head_pos_z": head_pos_z}

    df_cols = list(cols.values())
    if not all(col in df.columns for col in df_cols):
        raise Exception(f"DataFrame is missing some columns from <{df_cols}>")

    # Initialize matrices, results DF, window indices
    hz = frequencies(df[time])
    gaze_matrix, head_matrix = get_gaze_head_matrices(df, cols)
    fixation_cols = ["fixation", "fixation_start", "fixation_end", "fixation_duration"]
    fixation_df = pd.DataFrame(np.zeros(df.shape[0], len(fixation_cols), int),
                               columns=fixation_cols)
    final = data.shape[0] - 1
    window_start = 0

    # Find fixation windows
    while window_start < final:
        window_end = window_start + 1
        # Extend window until total window time exceeds the given minimum valid window time
        while (df[time].iloc[window_end] - df[time].iloc[window_start]) < min_window_time:
            window_end += 1
            if window_end > final:
                return fixation_df

        # Current window isn't a valid fixation, increment start
        if (not valid_frequences(hz[window_start:window_end+1])
            or not is_fixating(gaze_matrix, head_matrix, window_start, window_end)):
            window_start += 1
        else:
            # Extend the window while in a valid fixation
            while (is_fixating(gaze_matrix, head_matrix, window_start, window_end)
                   and window_end <= final):
                window_end += 1
            window_end -= 1  # decrement since we've exceeded the fixation window in last loop iteration
            # Process the fixation window
            duration = df[time].iloc[window_end] - df[time].iloc[window_start]
            fixation_df["fixation_start"].iloc[window_start] = 1
            fixation_df["fixation_end"].iloc[window_end] = 1
            fixation_df["fixation"].iloc[window_start:window_end+1] = 1
            fixation_df["fixation_duration"].iloc[window_end] = duration

            window_start = window_end

    return fixation_df


if __name__ == '__main__':

    # Prepare paths and directories
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    if not data_dir.exists():
        raise Exception("Data directory doesn't exist:", data_dir)
    if not output_dir.exists():
        raise Exception("Output directory doesn't exist:", output_dir)

    result_dir = f"disp_{args.angle_dispersion_threshold}_time_{args.min_window_time}_freq_{args.min_frequency}"
    result_dir = result_dir.replace('.', '_')

    print('##################################################')
    print('################ Running VR IDT ##################')
    print('##################################################')

    try:
        os.mkdir(output_dir / result_dir)
    except FileExistsError:
        print('Output directory already exists. May overwrite files.')
    result_dir = output_dir / result_dir
    print('Writing to:', result_dir)

    # Process all the data files in given directory
    for file in os.listdir(data_dir):
        if file.endswith('.csv'):
            print(f"{file}   ::   {datetime.now() - start}")
            # Run classification algo and add fixation labels
            df = vr_itd(data_dir / file,
                        args.min_window_time,
                        args.angle_dispersion_threshold,
                        args.min_frequency)
            df.to_csv(result_dir / file, index=False)

    print('\nDONE')
