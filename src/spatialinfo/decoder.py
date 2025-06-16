import warnings

import numpy as np

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
    y_mu = y - np.mean(data, axis=0)
    cov = np.cov(data.T)
    inv_covmat = np.linalg.inv(cov)
    left = np.dot(y_mu, inv_covmat)
    mahal = np.sqrt((np.dot(left, y_mu.T)))
    return mahal
