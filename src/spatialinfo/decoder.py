import warnings

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

from spatialinfo.spatial_information import avg_activity, binning

# function to create templates from calcium traces and positional data


def clean_data(dff: np.ndarray, behavior: np.ndarray):
    """
    Removes large downward jumps in bin values (teleport artifacts).

    Parameters:
        dff (np.ndarray): Feature matrix (time x features).
        behavior (np.ndarray): Behavior data (time x behavioral variables),
            where column 1 contains bin values.

    Returns:
        dff_clean (np.ndarray): Cleaned feature matrix.
        behavior_clean (np.ndarray): Cleaned behavior matrix.
    """
    # Identify where the bin values decrease by more than 1
    jump_id = np.where(np.diff(behavior[:, 1]) < -1)[0]  # Get index of jump
    # Create a boolean mask to exclude those indices and the following row
    mask = np.ones(len(behavior), dtype=bool)
    mask[jump_id] = False  # Remove the jump point
    mask[jump_id + 1] = False  # Also remove the following point

    # Apply mask to both datasets
    dff_clean = dff[mask, :]
    behavior_clean = behavior[mask, :]

    return dff_clean, behavior_clean


def create_templates(
    dff: np.ndarray, behavior: np.ndarray, n_bins: int = 30, average: bool = False
):
    """
    Averages population activity based on positional
    information to create templates.

    Parameters:
        dff (numpy.ndarray):
            A 2D array of delta(f)/f calcium traces,
            with features as columns and timepoints as rows
        bh (numpy.ndarray):
            A 2D array where:
            - column 1 contains X
            - column 2 contains Y
            - column 3 contains trial
            - rows are timepoints
        n_bins (int): number of spatial bins to sort the data into.
        average (boolean): in case the algorithm yields templates with
            the same positional ID, return average. Defaults to False.

    Returns:
        templates (numpy.ndarray): each column is one template
        positions (numpy.ndarray): column 1: spatial bin,
            column 2: corridor,
            row: indicates template column in templates
        bins (numpy.ndarray or IntervalIndex): returns the computed bins
    """
    # start with input checks

    # get bin values and digitize
    min_value, max_value = behavior[:, 1].min(), behavior[:, 1].max()
    bin_edges = np.linspace(min_value, max_value, (n_bins + 1))
    digit_array = np.digitize(
        behavior[:, 1], bin_edges, right=False
    )  # FIXME: np.append?
    # create unique ID for each bin by adding offset according to corridor:
    unique_bins = np.array(
        [
            bin_val + 30 if corridor != 0 else bin_val
            for bin_val, corridor in zip(digit_array, behavior[:, 0])
        ]
    )

    # Step 1: Find segment boundaries
    change_indices = (
        np.where(np.diff(unique_bins) != 0)[0] + 1
    )  # Find where bins change
    segment_splits = np.split(
        np.arange(len(unique_bins)), change_indices
    )  # Group indices by segment

    # Step 2: Compute averages for each segment
    averaged_features = np.array(
        [dff[segment].mean(axis=0) for segment in segment_splits]
    )

    # Step 3: Associate with unique bin values
    unique_bin_values = [
        unique_bins[segment[0]] for segment in segment_splits
    ]  # First value of each segment

    if average:
        pass  # FIXME

    return averaged_features.astype(float), unique_bin_values, bin_edges


def match_templates(
    templates: np.ndarray,
    positions: np.ndarray,
    input_vector: np.ndarray,
    distance_metric: str = "correlation",
):
    """
    Decodes the position based on the closest matching template.

    This function finds the most similar template to the `input_vector`
    using the specified `distance_metric` and returns the corresponding
    spatial bin and corridor.

    Parameters:
        templates (numpy.ndarray):
            A 2D array where each column represents a different template.
        positions (numpy.ndarray):
            A 1D array where:
            - Column 1 contains spatial bin indices.
            - Rows correspond to the columns in `templates`.
        input_vector (numpy.ndarray):
            A 1D array representing the input vector to be decoded based
            on similarity to `templates`.
        distance_metric (str, optional):
            The metric used to determine the closest match.
            Supported values are:
            - "correlation" (default)
            - "cosine"
            - "euclidean"
            - "mahalanobis"

    Returns:
        tuple: (spatial_bin, corridor), where:
            - spatial_bin (int): The estimated spatial bin
                based on the closest template.
            - corridor (int): The corresponding corridor.

    Raises:
        ValueError: If an unsupported `distance_metric` is provided.

    Example:
        >>> templates = np.random.rand(100, 5)  # 5 templates, 100 features
        >>> positions = np.array([[1, 0], [2, 0], [3, 1], [4, 1], [5, 1]])
        >>> input_vector = np.random.rand(100)
        >>> spatial_bin, corridor = match_templates(
        ...     templates, positions, input_vector, distance_metric="cosine"
        ... )
    """
    if distance_metric not in ["correlation", "cosine", "euclidean", "mahalanobis"]:
        raise ValueError(
            'unknown distance metric. Accepted values: "correlation", '
            '"cosine", "euclidean", "mahalanobis"'
        )

    # the input vector is stacked as the last column of the templates
    template_matrix = np.column_stack((templates.T, input_vector)).astype(float).T
    match distance_metric:
        case "correlation":
            return positions[np.argmax(np.corrcoef(template_matrix)[-1, :-1])]

        case "cosine":
            dot_product = np.linalg.multi_dot([input_vector, templates.T])
            magnitude = [
                np.linalg.norm(templates[i, :]) for i in range(np.shape(templates)[0])
            ]
            mag_input = np.sqrt(input_vector.dot(input_vector))
            return positions[
                np.argmax(dot_product / (np.array(magnitude).dot(mag_input)))
            ]

        case "euclidean":
            diff_matrix = np.array(
                [templates[i, :] - input_vector for i in range(np.shape(templates)[0])]
            )
            eucl_dist = [
                np.linalg.norm(diff_matrix[j, :])
                for j in range(np.shape(diff_matrix)[0])
            ]
            return positions[np.argmin(eucl_dist)]

        case "mahalanobis":
            if len(np.unique(positions)) == len(positions):
                warnings.warn(
                    "Provided templates do not support operations on distributions. "
                    "Output defaults to euclidean distance."
                )

            bin_mahal_output = np.zeros(np.shape(np.unique(positions)))

            for bin in np.unique(positions):
                template_distribution = templates[
                    (np.array(positions) == bin).nonzero(), :
                ][0].astype(float)
                bin_mahal = mahalanobis(input_vector, template_distribution)
                bin_mahal_output[bin - np.min(np.unique(positions))] = bin_mahal

            return np.argmin(bin_mahal_output) + 1


def mahalanobis(y=None, data=None):
    """Calculate the Mahalanobis distance between a point and a distribution.

    The Mahalanobis distance is a measure of the distance between a point P and the mean
    of a distribution D, scaled by the covariance matrix.
    It represents how many standard     deviations away P is from the mean of D.

    Parameters:
        y (numpy.ndarray): Point for which to calculate the Mahalanobis distance.
            Should be a 1-D array.
        data (numpy.ndarray): Dataset representing the distribution.
        Should be a 2-D array where rows are observations and columns are variables.

    Returns:
        float: The Mahalanobis distance between the point and the distribution.

    Raises:
        numpy.linalg.LinAlgError: If the covariance matrix is singular and
        cannot be inverted.

    Note:
        The function assumes the data is normally distributed and the covariance matrix
        is positive definite.
    """
    y_mu = y - np.mean(data, axis=0)
    cov = np.cov(data.T)
    inv_covmat = np.linalg.inv(cov)
    left = np.dot(y_mu, inv_covmat)
    mahal = np.sqrt((np.dot(left, y_mu.T)))
    return mahal


def fixed_template(dff, behavior, n_bins=30, n_neurons=5):
    """
    Implements direct basis decoding with LOOCV using fixed binning.

    Parameters:
        dff (pandas.DataFrame): Calcium imaging traces for all neurons.
        behavior (pandas.DataFrame): Behavioral data with X (corridor), Y (position).
        n_corridors (int): Number of corridors in the setting (default=2).
        n_bins (int): Number of spatial bins (default=30).
        n_neurons (int): Number of most active neurons to use for decoding (default=5).

    Returns:
        decoded_position (pandas.DataFrame): Decoded X and Y positions.
    """
    # Preprocess behavior data
    # bh = remove_interpolated_values(behavior, n_corr=n_corridors)
    # if "trial" not in bh.columns:
    #    bh = add_trial_column(bh)

    # Initialize storage for decoded results
    decoded_positions = []

    # Perform Leave-One-Out Cross-Validation (LOOCV)
    for trial in behavior["trial"].unique():
        print(f"Performing LOOCV on trial {trial}.")

        # Split data into training and test sets
        dff_test = dff[behavior["trial"] == trial]
        dff_train = dff[behavior["trial"] != trial]
        bh_test = behavior[behavior["trial"] == trial]
        bh_train = behavior[behavior["trial"] != trial]

        # Compute binning for training data
        time_per_bin, summed_traces, bins = binning(dff_train, bh_train, n_bins=n_bins)
        # Compute average activity templates (fixed decoder)
        avg_act_mtx = avg_activity(time_per_bin, summed_traces)

        # Apply the same binning to test data
        time_per_bin, summed_traces, _ = binning(
            dff_test, bh_test, n_bins=n_bins, bins=bins
        )
        loocv_bins = avg_activity(time_per_bin, summed_traces)

        # Standardize each neuron's activity across all bins and corridors
        standardized_act_mtx = avg_act_mtx.copy()
        standardized_loocv_bins = loocv_bins.copy()
        # neuron_stats = {}  # Store means and stds for later normalization
        for neuron in range(dff.shape[1]):
            neuron_mean = avg_act_mtx.loc[
                :, avg_act_mtx.columns.get_level_values("Neuron") == neuron
            ].values.mean()
            neuron_std = avg_act_mtx.loc[
                :, avg_act_mtx.columns.get_level_values("Neuron") == neuron
            ].values.std()
            standardized_act_mtx.loc[
                :, standardized_act_mtx.columns.get_level_values("Neuron") == neuron
            ] = (
                avg_act_mtx.loc[
                    :, avg_act_mtx.columns.get_level_values("Neuron") == neuron
                ]
                - neuron_mean
            ) / neuron_std
            standardized_loocv_bins.loc[
                :, standardized_loocv_bins.columns.get_level_values("Neuron") == neuron
            ] = (
                loocv_bins.loc[
                    :, loocv_bins.columns.get_level_values("Neuron") == neuron
                ]
                - neuron_mean
            ) / neuron_std

        # Get the single corridor value for this trial
        # Check if there's only one unique X value, otherwise take the most common one
        if len(bh_test["X"].unique()) == 1:
            corridor = bh_test["X"].unique()[0]
        else:
            corridor = bh_test["X"].mode()[0]

        # Loop through each space bin in the test trial
        for space_bin in range(n_bins):
            if space_bin not in standardized_loocv_bins.index:
                continue  # Skip if test data does not contain this bin

            # Extract the population vector for this bin
            pop_vector = standardized_loocv_bins.xs(
                key=corridor, level="Corridor", axis=1
            ).loc[space_bin]

            # Get the most active cells in this bin using standardized values
            top_neurons = pop_vector.nlargest(n_neurons)

            # Calculate relative activity (how much more active compared to others)
            relative_activity = top_neurons / top_neurons.sum()

            # Get corresponding activity maps from standardized templates
            scaled_matrix = standardized_act_mtx[top_neurons.index].multiply(
                relative_activity, axis=1, level="Neuron"
            )

            # Sum across neurons for each corridor separately
            decoded_map = pd.DataFrame()
            for corr in behavior["X"].unique():
                corridor_matrix = scaled_matrix.loc[
                    :, scaled_matrix.columns.get_level_values("Corridor") == corr
                ]
                decoded_map[corr] = corridor_matrix.sum(axis=1)

            # Get the most active bin and its corresponding corridor
            row_idx, col_idx = np.unravel_index(
                decoded_map.values.argmax(), decoded_map.shape
            )
            decoded_bin = int(decoded_map.index[row_idx])
            decoded_corridor = int(decoded_map.columns[col_idx])

            # Store results
            decoded_positions.append(
                {
                    "trial": trial,
                    "true_corridor": corridor,
                    "decoded_corridor": decoded_corridor,
                    "true_bin": space_bin,
                    "decoded_bin": decoded_bin,
                }
            )

    return pd.DataFrame(decoded_positions)


def proximity_weighted_accuracy(y_true, y_pred, max_distance=2):
    """
    Compute accuracy where predictions closer to the true bin get partial credit.

    Parameters:
        y_true (array-like): True bin labels.
        y_pred (array-like): Predicted bin labels.
        max_distance (int): Maximum distance for partial credit.

    Returns:
        float: Proximity-weighted accuracy score.
    """
    total_score = 0
    for true, pred in zip(y_true, y_pred):
        distance = abs(true - pred)
        if distance <= max_distance:
            total_score += 1 - (distance / (max_distance + 1))

    return total_score / len(y_true)


def build_keras_model(input_size, n_bins, n_corridors):
    inputs = tf.keras.Input(shape=(input_size,))

    # Shared layers
    x = layers.Dense(128, activation="relu")(inputs)  # 128
    x = layers.Dense(64, activation="relu")(x)  # 64

    # Two separate outputs
    bin_output = layers.Dense(n_bins, activation="softmax", name="bin_output")(x)
    corridor_output = layers.Dense(
        n_corridors, activation="softmax", name="corridor_output"
    )(x)

    model = models.Model(inputs=inputs, outputs=[bin_output, corridor_output])

    model.compile(
        optimizer="adam",
        loss={
            "bin_output": "sparse_categorical_crossentropy",
            "corridor_output": "sparse_categorical_crossentropy",
        },
        metrics={"bin_output": ["accuracy"], "corridor_output": ["accuracy"]},
    )  # Fix: Separate metrics

    return model
