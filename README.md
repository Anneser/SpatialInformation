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

### Code Formatting and Style

This project uses several tools to maintain consistent code style and quality:

1. **black** - Code formatter
   - We use Black as our Python code formatter with a line length of 88 characters
   - To format your code, run:
   ```bash
   black .
   ```

2. **isort** - Import sorter
   - isort automatically sorts and formats Python imports
   - To sort imports in your code, run:
   ```bash
   isort .
   ```

3. **flake8** - Code linter
   - Our flake8 configuration uses a max line length of 88 (to match Black)
   - Ignores specific rules (E203, W503) to be compatible with Black
   - To check your code for style issues:
   ```bash
   flake8
   ```

You can install these development tools with:
```bash
pip install black flake8 isort
```

We recommend setting up your editor to run these tools automatically on save. For VS Code, you can add these settings to your workspace settings:

```json
{
    "python.formatting.provider": "black",
    "editor.formatOnSave": true,
    "python.sortImports.args": ["--profile", "black"],
    "python.linting.flake8Enabled": true
}
```

