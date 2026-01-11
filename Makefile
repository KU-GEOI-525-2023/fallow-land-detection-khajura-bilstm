CONFIG ?= configs/data/dataset_v2.yaml
POETRY ?= poetry

ifeq ($(OS),Windows_NT)
	POETRY_CHECK = where $(POETRY) >NUL 2>&1
else
	POETRY_CHECK = command -v $(POETRY) >/dev/null 2>&1
endif

.PHONY: check-poetry
check-poetry:
	@$(POETRY_CHECK) || (echo "Poetry not found. Install it and ensure it is on PATH."; exit 1)

.PHONY: extract-train-data
extract-train-data: check-poetry
	$(POETRY) run python src/land_classifier/data/scripts/extract_training_data_v2.py --config $(CONFIG)

EDA_V5_CONFIG ?= configs/eda/v5.yaml
RULE_BASED_CONFIG ?= configs/baselines/rule_based.yaml
TRAIN_BASELINE_CONFIG ?= configs/train/baseline.yaml
TRAIN_V2_CONFIG ?= configs/train/v2_optimized.yaml
TRAIN_V3_CONFIG ?= configs/train/v3_stable.yaml
TRAIN_BASELINE_NAME := $(basename $(notdir $(TRAIN_BASELINE_CONFIG)))
TRAIN_V2_NAME := $(basename $(notdir $(TRAIN_V2_CONFIG)))
TRAIN_V3_NAME := $(basename $(notdir $(TRAIN_V3_CONFIG)))
INFER_BASELINE_CONFIG ?= configs/inference/baseline.yaml
INFER_V2_CONFIG ?= configs/inference/v2.yaml
INFER_V3_CONFIG ?= configs/inference/v3.yaml

.PHONY: eda-v5
eda-v5: check-poetry
	$(POETRY) run python notebooks/exploratory/eda_v5.py --config $(EDA_V5_CONFIG)

.PHONY: rule-based
rule-based: check-poetry
	$(POETRY) run python src/land_classifier/baselines/rule_based_land_classifier.py \
		--config $(RULE_BASED_CONFIG)
	$(POETRY) run python src/land_classifier/evaluation/evaluate_rule_based.py \
		--config $(RULE_BASED_CONFIG)

.PHONY: train-baseline
train-baseline: check-poetry
	$(POETRY) run python src/land_classifier/training/train.py train=$(TRAIN_BASELINE_NAME)

.PHONY: train-v2
train-v2: check-poetry
	$(POETRY) run python src/land_classifier/training/train.py train=$(TRAIN_V2_NAME)

.PHONY: train-v3
train-v3: check-poetry
	$(POETRY) run python src/land_classifier/training/train.py train=$(TRAIN_V3_NAME)

.PHONY: infer-baseline
infer-baseline: check-poetry
	$(POETRY) run python src/land_classifier/inference/inference.py --config $(INFER_BASELINE_CONFIG)

.PHONY: infer-v2
infer-v2: check-poetry
	$(POETRY) run python src/land_classifier/inference/inference.py --config $(INFER_V2_CONFIG)

.PHONY: infer-v3
infer-v3: check-poetry
	$(POETRY) run python src/land_classifier/inference/inference.py --config $(INFER_V3_CONFIG)
