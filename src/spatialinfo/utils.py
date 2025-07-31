import itertools
import math

import numpy as np
import pandas as pd


def binning_function(behavior, sec_per_bin=0.5, fps=30):
    """
    Bins occupancy and activity data into temporal bins.
    Assumes that distinct corridors are characterized by X position and
    that a trial column exists in behavior

    Parameters:
        dff (pandas.core.frame.DataFrame): df/f calcium imaging data
            (rows: time; columns: neurons)
        behavior (pandas.core.frame.DataFrame): contains X, Y columns with
            spatial information, trial column and frdIn with tail vigor
        sec_per_bin (int): how many seconds will be averaged into one bin,
            defaults to 0.5
        fps (int): recording speed in frames per second, defaults to 30

    Returns:
        activity_data (np.ndarray): activity matrix after binning,
            columns = features, rows = bins
        binned_data (DataFrame): average position & time of animal during each bin,
            columns = X, Y, trial, session and trial time, rows = bins
    """
    # Compute the number of frames per bin
    frames_per_bin = sec_per_bin * fps

    # Determine number of bins
    num_bins = int(np.ceil(len(behavior) / frames_per_bin))

    # Initialize lists to store binned data
    binned_spatial = []
    binned_x = []
    binned_y = []

    for i in range(num_bins):
        start_idx = math.floor(i * frames_per_bin)
        end_idx = math.ceil(min((i + 1) * frames_per_bin, len(behavior)))

        spatial_bin = behavior[["trial"]].iloc[start_idx:end_idx].mode(axis=0).values
        x_bin = behavior[["X"]].iloc[start_idx:end_idx].mode(axis=0).values
        y_bin = behavior[["Y"]].iloc[start_idx:end_idx].mean(axis=0).values

        if np.any(np.diff(behavior.trial.iloc[start_idx:end_idx].values) != 0):
            continue
        else:
            binned_spatial.append(spatial_bin[0][0])
            binned_x.append(x_bin[0])
            binned_y.append(y_bin[0])

    # Convert lists to numpy arrays
    trial_data = np.array(binned_spatial).astype(float)
    x_data = np.array(binned_x).astype(float)
    y_data = np.array(binned_y).astype(float)
    j = 0
    session_time = [j + i * 0.5 for i in range(len(trial_data))]
    trial_time = [
        i * 0.5
        for group in itertools.groupby(trial_data)
        for i, _ in enumerate(list(group[1]))
    ]

    binned_data = pd.DataFrame(
        {
            "X": [i[0] for i in x_data],
            "Y": y_data,
            "trial": trial_data,
            "session_time": session_time,
            "trial_time": trial_time,
        }
    )

    return binned_data
