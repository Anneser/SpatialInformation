# SpatialInformation


## Setup Instructions
### General Coding Setup

1. **Create and activate the virtual environment:**

    ```bash
    conda create --name SpatialInformation python=3.12.3
    conda activate SpatialInformation
    ```

2. **Install the required packages:**

    ```bash
    pip install --upgrade pip
    pip install tensorflow
    pip install pandas numpy matplotlib scipy seaborn pillow scikit-learn jupyter ipykernel
    python -m ipykernel install --user --name SpatialInformation --display-name "Python (SpatialInformation)"
    ```

    You can also recreate the environment by running

    ```bash
    conda env create --file environment.yml
    ```

3. **Install modules**
    ```bash
    pip install -e . 
    ```

