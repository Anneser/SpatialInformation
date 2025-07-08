Usage
=====

Installation
------------

To use `SpatialInformation`, first create a virtual environment and install the dependencies.

1. Create a virtual environment from `environment.yaml`

   If you are using **conda or mamba**, run:

   .. code-block:: console

      $ conda env create -f environment.yaml
      # or
      $ mamba env create -f environment.yaml

   Then activate the environment:

   .. code-block:: console

      $ conda activate SpatialInformation

2. Install the package in editable mode

   Once inside the environment, install the package:

   .. code-block:: console

      (spatialinformation) $ pip install -e .

General Usage
-------------

The package has been designed to analyze spatial information in neural data obtained from virtual reality experiments in adult zebrafish. The entry point would be a folder of raw data, containing the following files:

- *_behavior.pkl*: A pickle file containing the behavior data with X and Y columns. In our typical setting, Y corresponds to movement along a linear track and X encodes the corridor. Sampling is typically done at 30 fps.
- *_dff.pkl*: A pickle file containing the activity data, which is a 2D array with neurons as columns and time points as rows.

From here, you can use the functions in the `spatialinfo` module to analyze the data. The typical workflow has been designed in a Makefile, which can be found in the root directory. The Makefile provides a convenient way to run the analysis steps sequentially:

1. Preprocess data (add trial information, clean up behavior data, etc.).
2. Compute spatial information metrics (spatial tuning curves, place fields, etc.).
3. Perform dimensionality reduction (e.g., UMAP) and visualize the results.
4. Decode position based on activity data.