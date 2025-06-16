.PHONY: all clean preprocess stats decode manifold

# Define directories
DATA_DIR = data/processed_datasets
RESULTS_DIR = results
PREPROC_DIR = $(RESULTS_DIR)/preprocessed
STATS_DIR = $(RESULTS_DIR)/stats
DECODE_DIR = $(RESULTS_DIR)/decoding
MANIFOLD_DIR = $(RESULTS_DIR)/manifold

# Create all results directories
$(RESULTS_DIR)/% :
	if not exist "$@" mkdir "$@"

# Main target
all: preprocess stats decode manifold

# Preprocessing step
preprocess: $(PREPROC_DIR)
	python scripts/preprocessing.py --input $(DATA_DIR) --output $(PREPROC_DIR)

# Statistical analysis
stats: $(STATS_DIR) preprocess
	python scripts/stats.py --input $(PREPROC_DIR) --output $(STATS_DIR)

# Spatial decoding
decode: $(DECODE_DIR) preprocess
	python scripts/spatialdecoding.py --input $(PREPROC_DIR) --output $(DECODE_DIR)

# Manifold analysis
manifold: $(MANIFOLD_DIR) preprocess
	python scripts/manifold.py --input $(PREPROC_DIR) --output $(MANIFOLD_DIR)

# Clean all generated files
clean:
	if exist "$(RESULTS_DIR)" rmdir /s /q "$(RESULTS_DIR)"

# Create results structure
setup:
	if not exist "$(PREPROC_DIR)" mkdir "$(PREPROC_DIR)"
	if not exist "$(STATS_DIR)" mkdir "$(STATS_DIR)"
	if not exist "$(DECODE_DIR)" mkdir "$(DECODE_DIR)"
	if not exist "$(MANIFOLD_DIR)" mkdir "$(MANIFOLD_DIR)"
