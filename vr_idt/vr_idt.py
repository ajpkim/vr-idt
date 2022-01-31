import argparse
import math
import os

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
# Suppress pandas warning that we can safely ignore
pd.options.mode.chained_assignment = None

####################################################################################################
# Generate fixation/saccade classifications for VR data with time, eye tracking data in
# virtual world x, y, z space, and head x, y, z position. Function 'vr_itd' reads data
# from given csv and returns pd.DataFrame with additional cols for labels and hz (sampling rate).
# Classification algo works on data from single trial at a time (i.e. time from 0-600 for 10min trial).
#
# Necessary cols in csv files are:
# - 'time' (seconds)
# - 'gaze_world_x'
# - 'gaze_world_y'
# - 'gaze_world_z'
# - 'head_pos_x'
# - 'head_pos_y'
# - 'head_pos_z'
#
# Cols added to data are:
# - 'fix_class': where 1 is a fixation and 0 is a saccade
# - 'fix_start': 1 represents beginning of fixation
# - 'fix_stop': 1 represents fixation has ended with this row
# - 'fix_duration': last sample in fixation window provides total fixation duration
# - 'hz': the sampling rate based on previous row time difference
#
# Original paper: https://www.mdpi.com/1424-8220/20/17/4956
# Paper author code: https://github.com/ASAPLableni/VR-centred_I-DT_algorithm
#
# Run this script to process directory of csv data files. Must provide algo
# parameters as well as a location to write the output csv files. A new directory will be
# created in the given location '<output_dir>/time_<arg>_disp_<arg>_freq_<arg>.csv' where 'arg'
# is replaced with the params used. Outputted filenames match the input file names.
####################################################################################################

def add_frequencies_col(data) -> pd.DataFrame:
    """Add frequency col to given dataframe that measures the sampling rate based on adjacent sample times"""
    data['hz'] = -1
    # Handle the 0 index (can't compare idx 0 idx to idx -1)
    data['hz'].iloc[0] = 1 / data['time'].iloc[0]
    # Compute the sampling rate for each point i by comparing time at i vs. time at i-1
    for i in range(1, data.shape[0]):
        time_delta = data['time'].iloc[i] - data['time'].iloc[i-1]
        data['hz'].iloc[i] = 1/time_delta

    return data

def fixating(data, window_start_idx, window_end_idx, dispersion_threshold=1.5, min_freq=30) -> bool:
    """Return bool indicating whether window is part of a fixation"""
    window_data = data.iloc[window_start_idx:window_end_idx+1]

    if not all(hz > min_freq for hz in window_data['hz']):
        return False

    return check_window_dispersions(window_data, dispersion_threshold)

def scalar_product(x1, x2, y1, y2, z1, z2) -> float:
    """Compute the normalized scalar product in three dimensions by definition"""
    num = x1*x2 + y1*y2 + z1*z2
    den1 = np.sqrt(x1**2 + y1**2 + z1**2)
    den2 = np.sqrt(x2**2 + y2**2 + z2**2)

    return np.abs(num) / (den1*den2)

def check_window_dispersions(window_data, dispersion_threshold) -> bool:
    """Evaluate whether the dispersions within the window fall within fixation threshold"""
    x_diffs = np.array(window_data['gaze_world_x'] - window_data['head_pos_x'].mean())
    y_diffs = np.array(window_data['gaze_world_y'] - window_data['head_pos_y'].mean())
    z_diffs = np.array(window_data['gaze_world_z'] - window_data['head_pos_z'].mean())

    for i in range(window_data.shape[0]-1, 0, -1):
        dispersions = []
        for j in range(i):
            dispersions.append(scalar_product(x_diffs[i], x_diffs[j],
                                              y_diffs[i], y_diffs[j],
                                              z_diffs[i], z_diffs[j]))
        dispersions = np.clip(dispersions, -1, 1)  # Avoid floating point errors when taking acos
        dispersions = np.arccos(dispersions) * (180/np.pi)
        if any(disp > dispersion_threshold for disp in dispersions):
            return False

    return True

def vr_itd(csv_file, base_window_time=0.15, dispersion_threshold=1.50, min_freq=30) -> pd.DataFrame:
    """
    Fixation classification algorithm that reads data from given csv and classifies fixations vs saccades
    and provides additional cols to given DataFrame and implements the VR IDT algorithm as proposed
    in: https://www.mdpi.com/1424-8220/20/17/4956. Original code from authors
    is here: https://github.com/ASAPLableni/VR-centred_I-DT_algorithm.
    """
    data = pd.read_csv(csv_file)
    necessary_cols = ['time',
                      'gaze_world_x',
                      'gaze_world_y',
                      'gaze_world_z',
                      'head_pos_x',
                      'head_pos_y',
                      'head_pos_z',]
    if not all(col in data.columns for col in necessary_cols):
        raise Exception(f'Necessary columns: {necessary_cols} are not present in file: {csv_file}')

    # Prep data and initialize new cols, indices
    data = add_frequencies_col(data)
    data[['fix_class', 'fix_start', 'fix_end', 'fix_duration']] = 0
    final_idx = data.shape[0] - 1
    window_start_idx = 0

    while window_start_idx < final_idx:
        # Extend window until total window time >= base_window_time
        window_end_idx = window_start_idx + 1
        while (data['time'].iloc[window_end_idx] - data['time'].iloc[window_start_idx]) < base_window_time:
            window_end_idx += 1
            # Handle final window by marking window as saccade and returning
            if window_end_idx > final_idx:
                data['fix_class'].iloc[window_start_idx:] = 0
                break
        # Not currently in fixation, mark start as saccade and shift window start 1 step
        if not fixating(data, window_start_idx, window_end_idx, dispersion_threshold, min_freq):
            data['fix_class'].iloc[window_start_idx] = 0
            window_start_idx += 1
        else:
            # Extend the window as long as valid fixation holds
            while fixating(data, window_start_idx, window_end_idx, dispersion_threshold, min_freq) and window_end_idx <= final_idx:
                window_end_idx += 1
            # Begin processing the fixation window
            data['fix_start'].iloc[window_start_idx] = 1
            # Handle when trial ends on a fixation
            if window_end_idx > final_idx:
                data['fix_duration'].iloc[-1] = data['time'].iloc[-1] - data['time'].iloc[window_start_idx]
                data['fix_class'].iloc[window_start_idx:] = 1
            else:
            # Fixation has ended, mark window as fixation and last point as saccade
                data['fix_class'].iloc[window_start_idx:window_end_idx] = 1
                data['fix_class'].iloc[window_end_idx] = 0
                data['fix_end'].iloc[window_end_idx] = 1
                data['fix_duration'].iloc[window_end_idx-1] = data['time'].iloc[window_end_idx-1] - data['time'].iloc[window_start_idx]
            # Jump window start idx to current end idx + 1
            window_start_idx = window_end_idx + 1

    return data

if __name__ == '__main__':
    start = datetime.now()

    # Define command line args for algo params and data input/ouput
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--angle_dispersion_threshold', type=float, required=True, help='Max dispersion for fixation')
    parser.add_argument('-t', '--base_window_time', type=float, required=True, help='Starting window time length')
    parser.add_argument('-f', '--min_frequency', type=float, required=True, help='Minimum sampling frequency')
    parser.add_argument('-i', '--data_dir', type=str, required=True, help='Path of directory of data csv files to classify')
    parser.add_argument('-o', '--output_dir', type=str, required=True, help='Path of location to write to')
    args = parser.parse_args()

    # Prepare paths and directories
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    if not data_dir.exists():
        raise Exception("Data directory doesn't exist:", data_dir)
    if not output_dir.exists():
        raise Exception("Output directory doesn't exist:", output_dir)

    result_dir = f"disp_{args.angle_dispersion_threshold}_time_{args.base_window_time}_freq_{args.min_frequency}"
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
                        args.base_window_time,
                        args.angle_dispersion_threshold,
                        args.min_frequency)
            df.to_csv(result_dir / file, index=False)

    print('\nDONE')
