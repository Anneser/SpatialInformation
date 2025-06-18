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
$(PREPROC_DIR)/preprocess.done:
	python scripts/preprocessing.py --input $(DATA_DIR) --output $(PREPROC_DIR)
	echo.> $@

stats: $(STATS_DIR) $(PREPROC_DIR)/preprocess.done
	python scripts/stats.py --input $(PREPROC_DIR) --output $(STATS_DIR)
	echo.> $(STATS_DIR)/stats.done

# Spatial decoding
$(DECODE_DIR)/decode.done: $(PREPROC_DIR)/preprocess.done
	python scripts/spatialdecoding.py --input $(PREPROC_DIR) --output $(DECODE_DIR)
	echo.> $@

decode: $(DECODE_DIR) $(DECODE_DIR)/decode.done

# Manifold analysis
$(MANIFOLD_DIR)/manifold.done: $(PREPROC_DIR)/preprocess.done
	python scripts/manifold.py --input $(PREPROC_DIR) --output $(MANIFOLD_DIR)
	echo.> $@

manifold: $(MANIFOLD_DIR) $(MANIFOLD_DIR)/manifold.done

# Clean all generated files
clean:
	if exist "$(RESULTS_DIR)" rmdir /s /q "$(RESULTS_DIR)"

# Create results structure
setup:
	if not exist "$(PREPROC_DIR)" mkdir "$(PREPROC_DIR)"
	if not exist "$(STATS_DIR)" mkdir "$(STATS_DIR)"
	if not exist "$(DECODE_DIR)" mkdir "$(DECODE_DIR)"
	if not exist "$(MANIFOLD_DIR)" mkdir "$(MANIFOLD_DIR)"
